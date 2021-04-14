import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


def make_model(src_vocab, tgt_vocab, N=6, d_model=256, d_ff=512, h=4, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len=500)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_model, src_vocab)),
                           Generator(d_model, tgt_vocab), 
                           position,
                           d_model)
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, generator, position_enc, d_model, pad_idx=0):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.generator = generator
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.position_enc = position_enc

    def forward(self, src_seq, seq_time_idx=None, len_seq=None, trg_time_query=None, pass_fc=False):
        """
        src_seq : (n_batch x n_time_steps x n_events)
        seq_time_idx: (n_batch x n_time_steps x n_events)
        """
        n_b, n_seq, n_events = src_seq.size()

        # trg_io_mask : (n_b, n_seq, n_seq * n_events)
        src_mask, tgt_mask = self.get_masks(src_seq, device=src_seq.device)

        # expand last dimension into diagnoal matrix
        # (n_batch x n_seq x n_event) -> (n_batch x n_seq x n_event x n_event)
        src_seq = torch.diag_embed(src_seq)

        # merge dimension of event and time_steps
        # (n_batch x n_seq x n_events x n_events) -> (n_batch x [n_seq x n_events] x n_events)
        src_seq = src_seq.view(n_b, -1, n_events)   
        
        # (n_batch x [n_seq*n_events] x d_emb)
        zeros_inp = torch.zeros(n_b, (n_seq)*n_events , self.d_model).to(src_seq.device)
        inp_time_enc = self.position_enc(zeros_inp, n_seq, n_event=n_events)
        
        # (n_batch x n_seq x d_emb)
        zeros_trg = torch.zeros(n_b, (n_seq+1) , self.d_model).to(src_seq.device)
        pos_enc_trg = self.position_enc(zeros_trg, n_seq + 1)
        trg_time_enc = pos_enc_trg[:, 1:, :]

        enc_out = self.encode(src_seq, src_mask)
        dec_out = self.decode(enc_out, tgt_mask, len_seq)

        if not pass_fc:
            dec_out = self.generator(dec_out)

        return dec_out

    def encode(self, src_seq, src_mask):
        emb = self.src_embed(src_seq.long())
        return self.encoder(emb, src_mask)

    def decode(self, enc_out, tgt_mask, len_seq):
        output = self.decoder(enc_out, tgt_mask)
        max_seq_len = max(len_seq) # in exact number of steps 
        output = output[:, :max_seq_len, :]
        return output


    def get_masks(self, src_seq, device=None):
        # src_mask : (n_batch x n_seq x 1 x n_event)
        # fill 1s where actual input value exists
        src_mask = get_pad_mask(src_seq, self.pad_idx) 

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

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, n_seq, n_event=None):
        """
        x : (n_batch x (n_seq x n_event) x n_dim)
        pos_enc : (1 x n_seq x n_dim) -> (1 x (n_seq x n_event) x n_dim )
        """

        n_batch = x.size(0)
        pos_enc = self.pe[:, :n_seq].clone().detach()

        if n_event is not None:
            n_dim = pos_enc.size(-1)
            pos_enc = pos_enc.transpose(-1, -2).unsqueeze(-1).expand(1, n_dim, n_seq, n_event).reshape(1, n_dim, -1).transpose(-1, -2)
        return x + self.dropout(pos_enc)
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # return self.dropout(x)


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq_len, device=None):
    ''' For masking out the subsequent info. '''
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len), device=device), diagonal=1)).bool()
    
    # TEST to disable the subseq mask
    # subsequent_mask = (subsequent_mask*0+1).bool()

    return subsequent_mask

