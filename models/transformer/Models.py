''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
# print('__name__:{}'.format(__name__))

if __name__ == '__main__':
    from Layers import EncoderLayer, PredictorLayer
    from torch.nn.utils.rnn import pad_sequence
    
else:
    from models.transformer.Layers import EncoderLayer, PredictorLayer


"""
Code based on https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq_len, device=None):
    ''' For masking out the subsequent info. '''
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len), device=device), diagonal=1)).bool()
    
    # TEST to disable the subseq mask
    # subsequent_mask = (subsequent_mask*0+1).bool()

    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, n_seq, n_event=None):
        """
        x : (n_batch x (n_seq x n_event) x n_dim)
        pos_enc : (1 x n_seq x n_dim) -> (1 x (n_seq x n_event) x n_dim )
        """
        n_batch = x.size(0)
        pos_enc = self.pos_table[:, :n_seq].clone().detach()

        if n_event is not None:
            n_dim = pos_enc.size(-1)
            pos_enc = pos_enc.transpose(-1, -2).unsqueeze(-1).expand(1, n_dim, n_seq, n_event).reshape(1, n_dim, -1).transpose(-1, -2)
        return x + pos_enc


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=400, use_pos_enc=False):

        super().__init__()

        self.use_pos_enc = use_pos_enc
        self.src_word_emb = nn.Linear(n_src_vocab, d_word_vec, bias=False)
        self.src_temporal_emb = nn.Embedding(n_position, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_time, src_mask, n_seq=None, return_attns=False):
        '''
        src_seq : (n_batch x [seq_len x n_event] x n_event)
        src_time : (n_batch x [seq_len x n_event] x d_clock)
        '''
    
        enc_slf_attn_list = []
        
        # item_emb : (n_batch x [seq_len x n_event] x d_word_vec)
        item_emb = self.src_word_emb(src_seq)
        
        if self.use_pos_enc:
            item_emb = self.position_enc(item_emb, n_seq=n_seq, n_event=src_seq.size(-1))

        enc_output = self.dropout(item_emb)

        for enc_layer in self.layer_stack:
            
            enc_output, src_time, enc_slf_attn = enc_layer(enc_output, src_time, slf_attn_mask=src_mask)
            # enc_output, src_time, enc_slf_attn = torch.utils.checkpoint.checkpoint(enc_layer, enc_output, src_time, src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return (enc_output, enc_slf_attn_list)
        return enc_output


class Predictor(nn.Module):
    ''' A predictor model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, use_pos_enc=False):

        super().__init__()
        
        #NOTE: predictor use only one layer.
        n_layers = 1

        # self.trg_temporal_emb = nn.Embedding(n_position, d_word_vec, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            PredictorLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_output, subseq_io_mask, trg_time_query, src_time, len_seq, return_attns=False):

        pred_slf_attn_list = []

        # (n_b x n_seq x d_model)
        # trg_time_query = self.trg_temporal_emb(trg_time_query)

        for pred_layer in self.layer_stack:
            enc_output, enc_slf_attn = pred_layer(enc_output, trg_time_query, tmp_input=src_time, subseq_io_mask=subseq_io_mask)
            pred_slf_attn_list += [enc_slf_attn] if return_attns else []

        pred_output = self.layer_norm(enc_output)

        # cut the remaining padding elements in time-axis
        # (background: the input's time-axis is **all** number of event tokens
        #              the output's time-axis is **actual** number of time-steps)
        # self-attention with io-masking should be enough to summarizing infos 
        # across adequate temporal elements.

        max_seq_len = max(len_seq) # in exact number of steps 
        pred_output = pred_output[:, :max_seq_len, :]

        if return_attns:
            return (pred_output, pred_slf_attn_list)
        return pred_output        


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=500,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True, 
            use_pos_enc=False):

        super().__init__()

        self.clocks = list(range(1, 72)) + [1000]
        self.d_clock = len(self.clocks)

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout,
            use_pos_enc=use_pos_enc
            )

        self.predictor = Predictor(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.d_word_vec = d_word_vec

        self.softmax = torch.nn.Softmax(1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

        # weight tying
        # self.predictor.trg_temporal_emb.weight = self.encoder.src_temporal_emb.weight

    def get_masks(self, src_seq, device=None):
        # src_mask : (n_batch x n_seq x 1 x n_event)
        # fill 1s where actual input value exists
        src_mask = get_pad_mask(src_seq, self.src_pad_idx) 

        n_b, n_seq, n_events = src_seq.size()
        
        # (1 x n_seq x n_seq x 1)
        subseq_mask = get_subsequent_mask(n_seq, device).unsqueeze(-1)

        # (n_batch x n_seq x n_seq x n_event)
        src_mask = src_mask & subseq_mask

        # (n_batch x [n_seq x n_event] x [n_seq x n_event])
        src_mask = src_mask.reshape(n_b, n_seq, -1).transpose(-1,-2).unsqueeze(-1).expand(n_b, n_seq * n_events, n_seq, n_events).reshape(n_b, n_seq * n_events, -1)
        src_mask = src_mask.transpose(-1,-2)

        trg_io_mask = subseq_mask.expand(n_b, n_seq, n_seq, n_events).reshape(n_b, n_seq, n_seq * n_events)
        return src_mask, trg_io_mask

    def get_time_encoding(self, seq_time, clocks):
        clocks = torch.Tensor(clocks).to(seq_time.device)
        time_enc = seq_time.unsqueeze(-1).expand(list(seq_time.size()) + [len(clocks)])
        return torch.fmod(time_enc, clocks)


    def forward(self, src_seq, trg_seq=None, seq_time_idx=None,
                subseq_enc_mask=None, subseq_io_mask=None, 
                len_seq=None, trg_time_query=None, pred_step=False, pass_fc=False):

        """
        src_seq : (n_batch x n_time_steps x n_events)
        seq_time_idx: (n_batch x n_time_steps x n_events)
        """
        n_b, n_seq, n_events = src_seq.size()

        # trg_io_mask : (n_b, n_seq, n_seq * n_events)
        src_mask, trg_io_mask = self.get_masks(src_seq, device=src_seq.device)
        
        # expand last dimension into diagnoal matrix
        # (n_batch x n_seq x n_event) -> (n_batch x n_seq x n_event x n_event)
        src_seq = torch.diag_embed(src_seq)

        # merge dimension of event and time_steps
        # (n_batch x n_seq x n_events x n_events) -> (n_batch x [n_seq x n_events] x n_events)
        src_seq = src_seq.view(n_b, -1, n_events)
        
        """Prep Time Encoding

        Current version not uses multi-clock encoding. But use pos-enc (original paper) to use time-vector

        """

        # (n_batch x [n_seq x n_event] x d_clock)
        # seq_time_idx = self.get_time_encoding(seq_time_idx, self.clocks)
        # seq_time_idx = seq_time_idx.view(n_b, -1, self.d_clock)   


        # (n_batch x [n_seq*n_events] x d_emb)
        zeros_inp = torch.zeros(n_b, (n_seq)*n_events, self.d_word_vec).to(src_seq.device)
        inp_time_enc = self.position_enc(zeros_inp, n_seq=n_seq, n_event=n_events)
        
        # (n_batch x n_seq x d_emb)
        zeros_trg = torch.zeros(n_b, (n_seq+1) , self.d_word_vec).to(src_seq.device)
        pos_enc_trg = self.position_enc(zeros_trg, n_seq=n_seq + 1)
        trg_time_enc = pos_enc_trg[:, 1:, :]

        enc_output = self.encoder(src_seq, inp_time_enc, src_mask, n_seq=n_seq)

        """
        output whole time-steps (n_batch x n_seq x n_events)
        """

        seq_logit = self.predictor(enc_output, trg_io_mask, trg_time_enc, None, len_seq)

        if not pass_fc:
            seq_logit = self.trg_word_prj(seq_logit)

        return seq_logit


if __name__ == '__main__':

    # model param
    padding_idx = 0
    tf_d_word = tf_d_model = tf_d_k = tf_d_inner = 4
    num_layers = 4
    num_heads = 2
    dropout = 0.2
    use_pos_enc = False    

    # test data param
    n_batch, max_seq_len, event_size = 2, 4, 10
    window_size_y = 24

    transformer = Transformer(
                n_src_vocab=event_size, n_trg_vocab=event_size, 
                src_pad_idx=padding_idx, trg_pad_idx=padding_idx,
                d_word_vec=tf_d_word, d_model=tf_d_model, d_inner=tf_d_inner,
                n_layers=num_layers, n_head=num_heads, d_k=tf_d_k, d_v=tf_d_k, dropout=dropout, n_position=500,
                trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False,
                use_pos_enc=use_pos_enc,
                )
    
    # create data
    seq_events = torch.randint(0, 1, (n_batch, max_seq_len, event_size)).float()
    inp_times = torch.zeros(n_batch, max_seq_len, event_size)
    lengths = [max_seq_len] * n_batch

    # input time tensor
    nth_time_idx = torch.arange(1, max_seq_len + 1).unsqueeze(0).unsqueeze(-1) * window_size_y 
    nth_time_idx = nth_time_idx.expand(n_batch, max_seq_len, event_size) - torch.relu(inp_times)

    trg_time_query = [torch.Tensor(list(range(2, x + 2))).long() for x in lengths]
    trg_time_query = pad_sequence(
        trg_time_query, 
        batch_first=True, 
        padding_value=padding_idx
    )

    # run forward 
    output = transformer(seq_events, seq_time_idx=nth_time_idx, 
                    len_seq=lengths, trg_time_query=trg_time_query, pass_fc=False)
    
    print('input.size(): {}'.format(seq_events.size()))
    print('output: {}'.format(output))
    print('output.size(): {}'.format(output.size()))