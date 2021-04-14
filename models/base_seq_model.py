
import copy
import torch
import torch.nn as nn
import numpy as np
import itertools

from tabulate import tabulate
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .rnn_models import GRU, GRUCell, LSTM, LSTMCell
from .layer_lstm import LayerLSTM, LayerLSTMCell, HierRateOfChangeLSTM, \
    HierSparseTimeLSTM, GatedFeedbackLSTM
from .pytorch_ntm.ntm.aio import EncapsulatedNTM
from .transformer.Models import Transformer
from .transformer_thin.Models import Transformer as ThinTransformer
from .GRUD import GRUD
from .RETAIN import RETAIN
from .residual_seq_net import ResidualSeqNet
from .torch_transformer import TorchTransformerModel

# from .anno_transformer.Modules import make_model
from .regularizations.embed_regularize import embedded_dropout
from .regularizations.weight_drop import WeightDrop
from utils.tensor_utils import fit_input_to_output
# import torch.backends.cudnn as cudnn
# cudnn.enabled = True

import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

tabulate.PRESERVE_WHITESPACE = True
torch.manual_seed(5)


def fill_idx_multihot(mh_tensor, method='default', return_list=False):
        """
        Return a multihot vector with index value on last dimension axis
        :param vec: n_batch x max_seqlen x n_events
        :return: : list (  )
        """
        if method == 'default':
            mh_tensor = mh_tensor.long()
            for b in range(mh_tensor.size(0)):
                for s in range(mh_tensor.size(1)):
                    nz_idx = (mh_tensor[b][s] == 1).nonzero()
                    for n in nz_idx:
                        mh_tensor[b][s][n] = n
            return_val = mh_tensor.long()

        elif method == 'sparse':

            device = mh_tensor.device

            event_bch, time_bch = [], []
            for b in range(mh_tensor.size(0)):
                event_seq, time_seq = [], []
                for s in range(mh_tensor.size(1)):
                    nz_idx = (mh_tensor[b][s] == 1).nonzero()
                    nz_idx = nz_idx.squeeze().tolist()
                    nz_idx = [nz_idx] if type(nz_idx) == int else nz_idx
                    event_seq += nz_idx
                    time_seq += [s] * len(nz_idx)
                event_bch.append(event_seq)
                time_bch.append(time_seq)

            if not return_list:
                max_len = max(len(x) for x in event_bch)
                event_bch = [x + [0] * (max_len - len(x)) for x in event_bch]
                time_bch = [x + [0] * (max_len - len(x)) for x in time_bch]
                event_bch = torch.LongTensor(event_bch).to(device)
                time_bch = torch.LongTensor(time_bch).to(device)
                assert event_bch.size() == time_bch.size()
            return_val = (event_bch, time_bch)
        return return_val


def last_nonzero(arr, axis=0, invalid_val=-1):
    if len(arr.shape) > 2:
        return [last_nonzero(x, axis) for x in arr]
    arr = arr.T
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def take(input, index):
    out = torch.Tensor([[s_inp[s_idx] for s_inp, s_idx in zip(b_inp, b_idx)]
                        for b_inp, b_idx in zip(input, index)])
    if input.is_cuda:
        device = input.device
        out = out.to(device)
    return out

# nonlinear functions
sigmoid = nn.Sigmoid()
relu = nn.ReLU()
tanh = nn.Tanh()
logsoftmax = nn.LogSoftmax()
bceloss = nn.MultiLabelMarginLoss()


def masked_bce_loss(pred, trg, trg_len, use_bce_logit=False, use_stable=False, 
                    use_mse=False, pos_weight=None, event_weight=None):
    """
    :param pred: prediction tensor size of (n_batch x max_seq_len x target_size)
    :param trg:  target tensor size of (n_batch x max_seq_len x target_size)
    :param trg_len: target sequence sizes
    :return: loss value over batch elements
    """
    loss = torch.zeros(1, device=pred.device)
    eps = 1e-10        
    if use_bce_logit:
        if pos_weight is None:
            pos_weight = trg.sum(0).sum(0) / (trg.size(0) * trg.size(1))
        lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif use_mse:
        lossfn = nn.MSELoss()
    else:
        if event_weight is not None:
            lossfn = nn.BCELoss(reduction='none')
        else:
            lossfn = nn.BCELoss()

    while trg.dim() < pred.dim():
        trg = trg.unsqueeze(0)

    for a_batch_instance in zip(pred, trg, trg_len):
        probs_b, target_b, len_b = a_batch_instance

        if len_b == 1:
            probs = probs_b
            trgs = target_b
        else:
            probs = probs_b[:len_b]
            trgs = target_b[:len_b]

        if trgs.numel() == 1:
            continue

        probs = probs + eps if use_stable else probs
        trgs = trgs.to(probs.device)
        try:
            if event_weight is not None:
                loss += (lossfn(probs, trgs) * event_weight).sum()
            else:
                loss += lossfn(probs, trgs)
        except RuntimeError as e:
            raise e
        except ValueError as e:
            raise e

    return loss / len(trg_len)


class BaseMultiLabelLSTM(nn.Module):
    """
    input: multi-hot vectored sequence input
    output: multi-hot vectored sequence output
    """
    def __init__(self, event_size, window_size_y, target_size, hidden_dim, 
                 embed_dim,
                 use_cuda=False, batch_size=64, batch_first=True,
                 bidirectional=False, out_func='sigmoid',
                 rnn_type='GRU', num_layers=1, 
                 padding_idx=0,
                 pred_time=False,
                 output_type='plain',
                 device=None,
                 is_input_time=False,
                 hier_pred=False,
                 recent_bias=False,
                 memorize_time=False,
                 is_pack_seq=False,
                 target_type='multi',
                 pooling_type='mean',
                 dropout=0,
                 dropouti=0,
                 dropouth=0,
                 wdrop=0,
                 tie_weights=False,
                 pred_period=False,
                 pp_type='adaptive',
                 use_simple_gate=False,
                 use_orig_params=True,
                 pp_weight_scheme='default',
                 remap=False,
                 skip_hidden_state=False,
                 f_exp=False,
                 f_window=False,
                 rb_init="none",
                 manual_alpha=0,
                 inv_id_mapping=None,
                 clock_gate=False,
                 lab_pp_proc=False,
                 elapsed_time=False,
                 pp_merge_signal=None,
                 pp_concat=False,
                 rb_concat=False,
                 pp_ascounts=False,
                 num_heads=0,
                 use_pos_enc=False,
                 tf_d_word=16,
                 tf_d_model=16,
                 tf_d_inner=16,
                 tf_d_k=16,
                 tf_d_v=16,
                 attn_channel=None,
                 tf_by_step=False,
                 past_mem=False,
                 past_dist_lt=0,
                 past_dist_st=0,
                 past_as_count=False,
                 tf_pooling=False,
                 tf_type='full',
                 pm_softmax=False,
                 pred_future_steps=0,
                 tf_use_torch=False
                 ):
        super(BaseMultiLabelLSTM, self).__init__()

        self.event_size = event_size
        self.target_size = target_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embed_dim
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pooling_type = pooling_type

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_layer_input = nn.Dropout(dropouti)
        self.rnn_type = rnn_type
        self.out_func = out_func
        self.pred_time = pred_time
        self.output_type = output_type
        self.device = device
        self.target_type = target_type

        self.is_pack_seq = is_pack_seq
        self.padding_idx = padding_idx
        self.pred_period = pred_period
        self.pp_type = pp_type
        self.use_simple_gate = use_simple_gate
        self.use_orig_params = use_orig_params  # simpler method used for AIME

        self.past_mem = past_mem
        self.past_dist_lt = past_dist_lt
        self.past_dist_st = past_dist_st
        self.past_as_count = past_as_count
        self.pm_softmax = pm_softmax

        self.rb_init = rb_init
        self.manual_alpha = manual_alpha
        self.clock_gate = clock_gate
        self.drop = nn.Dropout(dropout)
        self.inv_id_mapping = inv_id_mapping
        self.lab_pp_proc = lab_pp_proc
        self.pp_concat = pp_concat
        self.rb_concat = rb_concat
        self.skip_hidden_state = skip_hidden_state
        self.pp_merge_signal = pp_merge_signal
        self.window_size_y = window_size_y
        self.hier_lstms = ['MyLayerLSTM', 'HierSparseTimeLSTM', 
                           'HierRateOfChangeLSTM', 'transformer-lstm', 
                           'transformer-sep-lstm']
        self.tf_by_step = tf_by_step
        self.tf_pooling = tf_pooling
        self.tf_type = tf_type
        self.tf_use_torch = tf_use_torch
        
        if rnn_type == 'GRU':
            self.is_pack_seq = True

        lstm_out_size = hidden_dim * num_layers \
            if (rnn_type in self.hier_lstms and not hier_pred) else hidden_dim
        if self.rnn_type == 'NTM':
            self.ntm = EncapsulatedNTM(
                        num_inputs=event_size, 
                        num_outputs=target_size,
                        controller_size=hidden_dim, 
                        controller_layers=num_layers, 
                        num_heads=1, N=128, M=20, device=device)
        else:
            self.ntm = None

        if self.rnn_type == 'CNN':
            # kernel determines how large days we apply conv1d operation 
            self.cnn_kernels = [2,4,8]
            self.convs = nn.ModuleDict(
                {f"{i}": nn.ModuleList(
                    [
                        nn.Conv1d(
                            in_channels=embed_dim, 
                            out_channels=embed_dim, 
                            kernel_size=k
                        ).to(device) for k in self.cnn_kernels
                    ]
                ) for i in range(self.num_layers)})
            self.pooling = nn.AdaptiveMaxPool1d(1)
            n_channels = len(self.cnn_kernels)
            self.fc_cnn = nn.Linear(n_channels * embed_dim, target_size)

        if self.past_mem:
            self.W_past_lt = nn.Linear(event_size, event_size)
            self.W_past_mt = nn.Linear(event_size, event_size)
            self.W_past_st = nn.Linear(event_size, event_size)

            self.W_past_lt.bias.data.fill_(0)
            self.W_past_mt.bias.data.fill_(0)
            self.W_past_st.bias.data.fill_(0)

            self.W_h = nn.Linear(hidden_dim, event_size)
            self.W_x = nn.Linear(event_size, event_size)
            self.W_z_lt = nn.Linear(event_size, event_size)
            self.W_z_mt = nn.Linear(event_size, event_size)
            self.W_z_st = nn.Linear(event_size, event_size)


        if self.rnn_type.startswith('transformer'):
            
            # self.transformer = make_model(
            #     src_vocab=event_size, 
            #     tgt_vocab=event_size, 
            #     N=self.num_layers, 
            #     d_model=tf_d_model, 
            #     d_ff=tf_d_model*2, 
            #     h=self.num_heads, 
            #     dropout=dropout).to(device)

            if self.tf_use_torch:
                self.transformer = TorchTransformerModel(
                    inp_ntoken=event_size, 
                    trg_ntoken=target_size, 
                    ninp=tf_d_inner,
                    nhead=self.num_heads, 
                    nhid=self.hidden_dim, 
                    nlayers=self.num_layers, 
                    dropout=self.dropout,
                )
            else:
                if self.tf_type == 'thin':
                    tf_model = ThinTransformer
                else:
                    tf_model = Transformer

                self.transformer = tf_model(
                    n_src_vocab=event_size, n_trg_vocab=target_size,
                    src_pad_idx=padding_idx, trg_pad_idx=padding_idx,
                    d_word_vec=tf_d_word, d_model=tf_d_model, d_inner=tf_d_inner,
                    n_layers=self.num_layers, n_head=self.num_heads, 
                    d_k=tf_d_k, d_v=tf_d_k, dropout=dropout, n_position=500,
                    trg_emb_prj_weight_sharing=False, 
                    emb_src_trg_weight_sharing=False,
                    use_pos_enc=use_pos_enc,
                    ).to(device)

                if self.rnn_type == 'transformer-lstm':
                    self.rnn = LayerLSTM(
                        cell_class=LayerLSTMCell,
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        dropout=dropout,
                        device=self.device)

            
        self.recent_bias = recent_bias
        if recent_bias or rb_concat:
            self.r_bias_linear = nn.Linear(event_size, target_size)

            if "xavier" in rb_init:
                torch.nn.init.xavier_uniform(self.r_bias_linear.weight)

            # NOTE: initialization based on prior is executed in trainer.py

        if self.clock_gate:
            self.clock_gate_layer1 = torch.nn.Linear(event_size * 3, target_size)
            self.clock_gate_layer2 = torch.nn.Linear(target_size, target_size)

        self.memorize_time = memorize_time
        if memorize_time:
            self.avg_time_linear = nn.Linear(event_size, event_size)
            self.rec_time_linear = nn.Linear(event_size, event_size)
            mem_time_avg = torch.zeros(event_size)
            self.mem_time_rec = torch.zeros(event_size) - 1
            self.mem_time_rec = torch.stack([self.mem_time_rec] * batch_size)

            if use_cuda:
                self.avg_time_linear = self.avg_time_linear.to(device)
                self.rec_time_linear = self.rec_time_linear.to(device)
                self.mem_time_rec = self.mem_time_rec.to(device)
                mem_time_avg = mem_time_avg.to(device)
            self.mem_time_avg = nn.Parameter(mem_time_avg)

        self.is_input_time = is_input_time

        # set fc_out layer
        if tie_weights:
            self.fc_out = lambda x: torch.nn.functional.linear(
                            x, self.embed_input.weight.t())
        else:
            layer_input_size = 0 if (skip_hidden_state or self.rnn_type == 'NoRNN') \
                else lstm_out_size
            
            if attn_channel == 'event':
                layer_input_size = event_size

            if self.rnn_type == 'transformer':
                layer_input_size = tf_d_model

            if self.pp_merge_signal in ['prior-only', 'seq-only']:
                mult_factor = 1
            else:
                mult_factor = 2


            if self.pp_concat:
                layer_input_size += target_size * mult_factor
            
            if self.rb_concat:
                layer_input_size += target_size

            if self.rnn_type == 'transformer-sep-lstm':
                layer_input_size += target_size

            if layer_input_size > 0:
                self.fc_out = nn.Linear(layer_input_size, target_size)
            else:
                self.fc_out = None

        if self.rnn_type not in ['transformer', 'transformer-lstm'] or self.tf_pooling:
            if target_type == 'multi':
                self.embed_input = nn.Linear(event_size, embed_dim, bias=False)
            elif target_type == 'single':
                self.embed_input = nn.Embedding(event_size, embed_dim,
                                                padding_idx=padding_idx,
                                                sparse=False)
        else:
            self.embed_input = None
            
        self.tie_weights = tie_weights
            
        self.init_weights(padding_idx=padding_idx)

        self.hier_pred = hier_pred

        if pred_time:
            self.fc_time_pred = nn.Linear(lstm_out_size, target_size)

        rnn_input_size = embed_dim + 1 if (pred_time and is_input_time) else embed_dim

        if self.rnn_type == 'MyGRU':
            self.rnn = GRU(cell_class=GRUCell,
                           input_size=rnn_input_size,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           dropout=dropout)
            if wdrop:
                self.rnn = WeightDrop(self.rnn, 
                    ['cell_{}'.format(i) for i in range(num_layers)], 
                    dropout=wdrop, nested_weights='u_h')

        elif self.rnn_type == 'MyLSTM':
            self.rnn = LSTM(cell_class=LSTMCell,
                            input_size=rnn_input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)
            if wdrop:
                self.rnn = WeightDrop(self.rnn, 
                    ['cell_{}'.format(i) for i in range(num_layers)], 
                    dropout=wdrop, nested_weights='weight_hh')

        elif (self.rnn_type in self.hier_lstms) \
                and self.rnn_type not in ['transformer-lstm']:

            if self.rnn_type == 'HierRateOfChangeLSTM':
                cell = HierRateOfChangeLSTM

            elif self.rnn_type == 'HierSparseTimeLSTM':
                cell = HierSparseTimeLSTM
            else:
                cell = LayerLSTM

            self.rnn = cell(cell_class=LayerLSTMCell,
                            input_size=rnn_input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            device=self.device)

            if wdrop:
                self.rnn = WeightDrop(self.rnn, ['cell_{}'.format(i) \
                    for i in range(num_layers)], dropout=wdrop, nested_weights='weight_hh')

        elif self.rnn_type == 'GFLSTM':
            self.rnn = GatedFeedbackLSTM(
                input_size=rnn_input_size, 
                hidden_size=hidden_dim, 
                num_layers=num_layers, 
                batch_first=batch_first,
                dropout=dropout
            )

        elif self.rnn_type != 'NoRNN' and (self.rnn_type not in \
            ['NTM', 'transformer', 'transformer-lstm', 'retain', 'CNN']):
            if self.rnn_type == 'GRU':
                self.rnn_cell = nn.GRU
            elif self.rnn_type == 'RNN':
                self.rnn_cell = nn.RNN
            elif self.rnn_type == 'LSTM':
                self.rnn_cell = nn.LSTM

            # PyTorch's RNN, GRU, LSTM 
            self.rnn = self.rnn_cell(input_size=rnn_input_size,
                                     hidden_size=hidden_dim,
                                     batch_first=batch_first,
                                     bidirectional=bidirectional,
                                     num_layers=num_layers,
                                     dropout=dropout)
            if wdrop:
                self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=wdrop)

        elif self.rnn_type == 'GRUD':
            self.rnn = GRUD(
                input_size=rnn_input_size,
                cell_size=hidden_dim,
                hidden_size=hidden_dim,
                X_mean=None, output_last=False
            )

        elif self.rnn_type == 'retain':
            self.rnn = RETAIN(dim_input=event_size, dim_emb=embed_dim, 
                dropout_input=self.dropout, dropout_emb=self.dropout, 
                dim_alpha=1, dim_beta=hidden_dim,
                dropout_context=self.dropout, dim_output=hidden_dim, batch_first=True)


        if output_type == 'hmlstm':
            self.fc_w_l = nn.Linear(lstm_out_size, num_layers)
            self.fc_w_l_time = nn.Linear(lstm_out_size, num_layers)

            for layer in range(num_layers):
                fc_out = nn.Linear(hidden_dim, target_size)
                setattr(self, 'fc_out_{}'.format(layer), fc_out)

                fc_out_time = nn.Linear(hidden_dim, target_size)
                setattr(self, 'fc_out_time_{}'.format(layer), fc_out_time)


    def get_fc_out(self, layer):
        return getattr(self, 'fc_out_{}'.format(layer))

    def get_fc_out_time(self, layer):
        return getattr(self, 'fc_out_time_{}'.format(layer))

    def forward(self, seq_events, lengths, hidden, 
            trg_times=None, inp_times=None, return_hidden_seq=False
        ):
        """
        inputs: n_batch x seqlen x 1
        """

        # Get Embedding and prep sequence
        
        if type(seq_events) == torch.Tensor:

            if self.target_type == 'multi':
                seq_events = seq_events.float()
            else:
                seq_events = seq_events.long()

            if self.rnn_type not in ['transformer', 'transformer-lstm', 'retain'] \
                    or self.tf_pooling:
                input_seq = self.embed_input(seq_events)
                # embed : n_batch x seqlen x 1

                input_seq = self.dropout_layer_input(input_seq)
            else:
                input_seq = seq_events # None

        if self.pred_time and self.is_input_time:
            time_seq = trg_times.unsqueeze(2)
            input_seq = torch.cat((input_seq, time_seq), dim=2)

        if self.rnn_type in ['GRU', 'RNN', 'LSTM'] and self.is_pack_seq:
            input_seq = pack_padded_sequence(input_seq, lengths,
                                             batch_first=True)

        # Run RNN

        _output = None
        plain_output = None

        if self.rnn_type in ['GRU', 'RNN', 'MyGRU']:
            _output, hidden = self.rnn(input_seq, hidden)
        elif self.rnn_type in ['LSTM']:
            if len(hidden[0].size()) < 3:
                hidden[0] = hidden[0].unsqueeze(0)
                hidden[1] = hidden[1].unsqueeze(0)
            _output, (h_n, c_n) = self.rnn(input_seq, hidden)
            hidden = (h_n, c_n)
        elif self.rnn_type in ['MyLSTM']:
            _output, (h_n, c_n) = self.rnn(input_seq, hidden, lengths)
            hidden = (h_n, c_n)
        elif self.rnn_type in self.hier_lstms \
                and self.rnn_type not in ['transformer', 'transformer-lstm', 'NTM']:
            _output, hidden = self.rnn(input_seq, hidden, lengths)
        
        elif self.rnn_type == 'GFLSTM':
            _output, hidden = self.rnn(input_seq, hidden, lengths)

        elif self.rnn_type in ['NTM']:
            n_batch, time_steps, _ = input_seq.size()
            self.ntm.init_sequence(batch_size=n_batch)
            output_steps = []
            for step in range(time_steps):
                input_step = seq_events[:, step, :].squeeze(1)
                output_step, _ = self.ntm(input_step)
                output_steps.append(output_step, )

            _output = torch.stack(output_steps, dim=1).to(self.device)

        elif self.rnn_type == "CNN":
            """
            based on 
            http://www.cse.chalmers.se/~richajo/nlp2019\
            /l2/Text%20classification%20using%20a%20word-based%20CNN.html
            permute input dim (batch, seq_len, embed_dim) to (batch, embed_dim, seq_len)
            to make sure that we convolve over the last dimension (seq_len)
            """
            preds = []

            for i in range(input_seq.size(1)):
                
                step_seq = input_seq[:, :i + 1]


                for idx, conv_layer in enumerate(self.convs.values()):

                    nb, ns, ne = step_seq.size()
                    # add padding for shorter sequence than max kernel
                    if ns < max(self.cnn_kernels):
                        len_need = max(self.cnn_kernels) - ns
                        pad = torch.zeros(nb, len_need, ne).to(self.device)
                        step_seq = torch.cat((pad, step_seq), dim=1)

                    x = step_seq.permute(0, 2, 1)
                    
                    conv_maps = [torch.relu(conv(x)) for conv in conv_layer]
                    pooled = [self.pooling(conv_map) for conv_map in conv_maps]
                    # merge all different conv kernels into single one
                    all_pooled = torch.cat(pooled, 1)
                    all_pooled = all_pooled.squeeze(2)
                    step_seq = self.dropout_layer(all_pooled)
                    if idx < self.num_layers - 1:
                        step_seq = step_seq.reshape(nb, -1, ne)

                pred_step = self.fc_cnn(step_seq)
                preds.append(pred_step)
            plain_output = torch.stack(preds, dim=1)

        if self.rnn_type.startswith('transformer'):    
            n_batch = len(seq_events)
            n_seq = max(len(x) for x in seq_events)
            n_event = seq_events.size(2)
            
            # input time tensor
            nth_time_idx = torch.arange(1, n_seq + 1)\
                .unsqueeze(0).unsqueeze(-1).to(self.device) * self.window_size_y 
            nth_time_idx = nth_time_idx.expand(n_batch, n_seq, n_event) - torch.relu(inp_times)
            
            trg_time_query = [
                torch.Tensor(list(range(2, x + 2))).long() for x in lengths
            ]
            trg_time_query = pad_sequence(
                trg_time_query, 
                batch_first=self.batch_first, 
                padding_value=self.padding_idx
            ).to(self.device)

            if self.tf_by_step:
                # step by step 
                for step in range(n_seq):
                    seq_step = seq_events[:, :step + 1]
                    nth_time_idx_step = nth_time_idx[:, :step + 1]
                    transformer_output = self.transformer(seq_step, seq_time_idx=nth_time_idx, 
                        len_seq=lengths, trg_time_query=trg_time_query, 
                        pass_fc=(self.rnn_type == 'transformer-lstm' or self.pp_concat))

            else:

                if self.tf_use_torch:
                    transformer_output = self.transformer(seq_events)
                else:
                    # NOTE: 'seq_time_idx' : input contains time step info about.. 
                    # (e.g., how much time before the window ends)
                    transformer_output = self.transformer(seq_events, seq_time_idx=nth_time_idx, 
                        len_seq=lengths, trg_time_query=trg_time_query, 
                        pass_fc=(self.rnn_type == 'transformer-lstm' or self.pp_concat))
                
            if self.rnn_type == 'transformer-lstm':
                _output, hidden = self.rnn(plain_output, hidden, lengths)

            if self.rnn_type == 'transformer':
                plain_output = transformer_output

        if self.rnn_type == 'retain':
            _output = []
            for step in range(1, seq_events.size(1) + 1):
                seq_step = seq_events[:, :step]
                length_step = [min(x, step) for x in lengths]
                _output_step, alphas, betas = self.rnn(seq_step, length_step)
                _output.append(_output_step)
            
            _output = torch.stack(_output, dim=1)

        if self.rnn_type in ['GRU', 'RNN', 'LSTM'] and self.is_pack_seq:
            _output, output_lengths = pad_packed_sequence(_output,
                                                     batch_first=self.batch_first,
                                                     total_length=max(lengths))

        # self.hier_lstms and
        if self.rnn_type != 'NoRNN' \
                and self.rnn_type not in ['NTM', 'transformer', 'retain'] \
                and self.rnn_type in self.hier_lstms and self.batch_first:
            _output = _output.transpose(0, 1)

        if return_hidden_seq:
            hidden = _output

        # Past Memory (long-term explicit) Computation
        if self.past_mem:

            # compute ranges
            # past_dist_Xt (by hour) -> lt_end (by num. of windows)
            lt_w_size = int(self.past_dist_lt / self.window_size_y) 
            st_w_size = int(self.past_dist_st / self.window_size_y)
            
            past_mems_lt = []
            past_mems_mt = []
            past_mems_st = []
            for step in range(seq_events.size(1)):

                lt_end = max(step - lt_w_size, 0)
                st_start = max(step - st_w_size, 0)

                lt_mem = seq_events[:, :lt_end].sum(1)
                mt_mem = seq_events[:, lt_end:st_start].sum(1)
                st_mem = seq_events[:, st_start:step].sum(1)
                
                if not self.past_as_count:
                    lt_mem = lt_mem > 0
                    mt_mem = mt_mem > 0
                    st_mem = st_mem > 0

                past_mems_lt.append(lt_mem)
                past_mems_mt.append(mt_mem)
                past_mems_st.append(st_mem)

            # ----------------------------------------------
            # # Logic test code
            # seq_len = 20
            # lt_w_size = 8
            # st_w_size = 1

            # seq = list(range(seq_len))
            # for step in range(seq_len):
            #     lt_end = max(step - lt_w_size, 0)
            #     st_start = max(step - st_w_size, 0)
            #     print('cur step: {}'.format(step))
            #     print('lt: {}'.format(seq[:lt_end]))
            #     print('mt: {}'.format(seq[lt_end:st_start]))
            #     print('st: {}'.format(seq[st_start:step]))
            #     print('-'*16)
            # ----------------------------------------------

            past_mems_lt = torch.stack(past_mems_lt, dim=1).float()
            past_mems_mt = torch.stack(past_mems_mt, dim=1).float()
            past_mems_st = torch.stack(past_mems_st, dim=1).float()

            # past_hist : n_batch x n_seq x n_events -> n_batch x n_seq x n_events
            past_hist_lt = torch.tanh(self.W_past_lt(past_mems_lt))
            past_hist_mt = torch.tanh(self.W_past_mt(past_mems_mt))
            past_hist_st = torch.tanh(self.W_past_st(past_mems_st))

            # # element-wise multiplication (hidden state x projected memory) 
            # # and sum over last dimension, then apply sigmoid -> make them a score
            # weight = torch.sigmoid((_output.squeeze(2) * past_hist).sum(-1)).unsqueeze(-1)

            # # multiply past memory with the weight
            # past_mems = weight * past_hist
            # # dim: n_batch x n_seq x hidden_dim

            # _output = _output + past_mems.unsqueeze(2)

        # Create RNN output


        if self.rnn_type != 'NoRNN' and self.rnn_type not in ['NTM', 'transformer', 'CNN']:
            # dropout on output of RNN/LSTM output hidden states
            _output = self.dropout_layer(_output)

        if self.output_type == 'hmlstm':
            # TODO: FIX contiguous

            l_weight = self.fc_w_l(_output.contiguous().view(
                _output.size(0), _output.size(1), -1))

            l_output = []
            for l in range(self.num_layers):
                fc_out = self.get_fc_out(l)
                l_output.append(fc_out(_output[:, :, l, :]))
            l_output = torch.stack(l_output, dim=-1)
            l_output = l_output.transpose(2, 3)
            w_output = l_weight.unsqueeze(-1).expand_as(l_output) * l_output
            plain_output = w_output.sum(2)

        elif self.rnn_type in self.hier_lstms and self.hier_pred:
            # CS3750:
            # n_batch x maxseqlen x num_layers x hidden_dim
            # TODO: FIX contiguous
            hidden_lower = _output.contiguous()[:,:,0,:]
            hidden_upper = _output.contiguous()[:,:,1,:]

            plain_output = self.fc_out(hidden_lower)
            # plain_output = self.fc_out(_output.contiguous().view(
            #     _output.size(0), _output.size(1), -1))

        if self.rnn_type == 'GFLSTM':
            if self.pp_concat:
                plain_output = _output
            else:    
                plain_output = self.fc_out(_output)

        if self.rnn_type != 'NoRNN' and self.rnn_type not in ['NTM', 'transformer', 'CNN']:
            if self.rnn_type in self.hier_lstms:
                _output = _output.contiguous().view(_output.size(0), _output.size(1), -1)
                if self.pp_concat or self.rb_concat or self.rnn_type == 'transformer-sep-lstm':
                    # NOTE: concatenate periodicity signals with lstm output
                    #       to make input for fc_out layer
                    # actual fc layer is applied after concatnation with lstm output
                    plain_output = _output
                else:
                    plain_output = self.fc_out(_output)
            else:
                plain_output = self.fc_out(_output)

        if self.rnn_type == 'transformer-sep-lstm':
            plain_output = torch.cat((plain_output, transformer_output), dim=2)
            if not self.pred_period:
                plain_output = self.fc_out(plain_output)

        # Periodicity Modeling Stuff
        if self.pred_period:
            self.pp.initialize_batch(
                input_seq.size(0), 
                self.event_size if self.inv_id_mapping is None else self.target_size,
                device=self.device
            )
            pp_steps = []
            pp_out = None
            max_time_step = seq_events.size(1)

            for t in range(1, max_time_step + 1):
                
                seq_step = seq_events[:, t-1, :]

                if inp_times is not None:
                    time_step = inp_times[:, t-1, :]

                if self.inv_id_mapping is not None:
                    seq_step = fit_input_to_output(
                        seq_step, self.inv_id_mapping.keys(), dim=1)

                    if inp_times is not None:
                        time_step = fit_input_to_output(
                            time_step, self.inv_id_mapping.keys(), dim=1)
                    else:
                        time_step = None
                        
                pp_out = self.pp(seq_step, t, time_step)
                pp_steps.append(pp_out)

            pp_preds = torch.stack(pp_steps, dim=1).to(self.device)
            
            # if self.inv_id_mapping is not None:
            #     pp_preds = fit_input_to_output(pp_preds, self.inv_id_mapping.keys())

            apply_masking = True

            if self.use_orig_params:
                if plain_output.dim() == 4 and plain_output.size(2) == 1:
                    plain_output = plain_output.squeeze(2)

                plain_output += self.pp_weight * pp_preds
    
            elif self.pp_concat:
                
                if self.rb_concat:
                    # AIM-Journal version
                    rb_signal = self.r_bias_linear(seq_events.float())
                    plain_output = torch.cat((plain_output, rb_signal), dim=2)

                if self.skip_hidden_state or plain_output is None:
                    plain_output = pp_preds
                    apply_masking = False
                else:
                    plain_output = torch.cat((plain_output, pp_preds), dim=2)
                    
                if apply_masking:
                    self.pp.event_mask = self.pp.event_mask.to(self.device)
                    self.fc_out.weight.data.mul_(self.pp.event_mask)

                if self.fc_out is not None:
                    plain_output = self.fc_out(plain_output)

            else:
                if self.manual_alpha > 0:
                    alpha = self.manual_alpha
                elif self.clock_gate:
                    w_infos = torch.cat(
                        (self.pp.pp.ri, self.pp.pp.ec, self.pp.pp.ac), dim=1)
                    h_w_infos = F.relu(self.clock_gate_layer1(w_infos))
                    alpha = F.sigmoid(self.clock_gate_layer2(h_w_infos))
                    alpha = alpha.unsqueeze(1)
                else:
                    pp_gate = self.pp_gate(plain_output, pp_preds)
                    alpha = F.sigmoid(pp_gate)

                plain_output = plain_output + alpha * pp_preds
            
            

        if not self.pred_period and self.rb_concat:

            rb_signal = self.r_bias_linear(seq_events.float())
            plain_output = torch.cat((plain_output, rb_signal), dim=2)
            plain_output = self.fc_out(plain_output)

        # Past Memory (long-term explicit) Addition
        if self.past_mem:

            if _output is None:
                # No LSTM
                _output = seq_events
                q = self.W_x(seq_events)
                plain_output = 0
            else:

                q = self.W_h(_output) + self.W_x(seq_events)

            gate_lt = torch.sigmoid(q + self.W_z_lt(past_hist_lt))
            gate_mt = torch.sigmoid(q + self.W_z_mt(past_hist_mt))
            gate_st = torch.sigmoid(q + self.W_z_st(past_hist_st))

            if self.pm_softmax:
                lms = torch.stack([gate_lt, gate_mt, gate_st])
                lms = F.softmax(lms, dim=-1)
                gate_lt, gate_mt, gate_st = torch.unbind(lms, dim=0)

            plain_output = plain_output \
                + (gate_lt * past_hist_lt) \
                    + (gate_mt * past_hist_mt) \
                        + (gate_st * past_hist_st)

        if self.recent_bias and not self.rb_concat:
            if plain_output is None:
                plain_output = self.r_bias_linear(seq_events.float())
            else:
                plain_output = plain_output + self.r_bias_linear(seq_events.float())

        assert (plain_output != plain_output).sum() == 0, "NaN!"

        time_output = None
        return plain_output, time_output, hidden

    def pp_gate(self, preds, periodic_pred):
        if self.use_simple_gate:
            w_preds = preds
            w_pp = periodic_pred
        else:
            w_preds = F.tanh(self.W_pp_gate_preds(preds))
            w_pp = F.tanh(self.W_pp_gate_pp(periodic_pred))
        w_concat = torch.cat((w_preds, w_pp), dim=2)
        gate = F.sigmoid(self.W_pp_gate(w_concat))
        return gate


    def _update_recent_time_mem(self, time_vec):
        use_cuda = time_vec.is_cuda
        device = time_vec.device if use_cuda  else None
        mem_time_rec = self.mem_time_rec.unsqueeze(1)

        if time_vec.size(0) == 1:
            mem_time_rec = mem_time_rec[0]

        time_vec = torch.cat((mem_time_rec, time_vec), dim=1)

        last_nz_idx = last_nonzero(time_vec.cpu().numpy(), 1)
        last_nz_idx = torch.LongTensor(last_nz_idx)
        if use_cuda:
            device = time_vec.device
            last_nz_idx = last_nz_idx.to(device)

        self.mem_time_rec = take(time_vec.transpose(1,2), last_nz_idx)

    def init_hidden(self, batch_size=None, device=None):
        init = 0.1
        if device is None:
            device = self.device
            
        if not batch_size:
            batch_size = self.batch_size

        if self.rnn_type in self.hier_lstms:
            hx = []
            for x in range(self.num_layers):

                h0 = torch.randn(batch_size, self.hidden_dim)
                c0 = torch.randn(batch_size, self.hidden_dim)
                h0.data.uniform_(-init, init)
                c0.data.uniform_(-init, init)

                h0, c0 = h0.to(device), c0.to(device)

                hx.append((h0, c0))

        else:
            if self.rnn_type in ['MyLSTM', 'MyGRU']:
                h0 = torch.randn(batch_size, self.hidden_dim)
                c0 = torch.randn(batch_size, self.hidden_dim)

            elif self.rnn_type == 'GFLSTM':
                h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
                c0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)

            else:
                h0 = torch.randn(self.num_layers * self.num_directions,
                                batch_size, self.hidden_dim)
                c0 = torch.randn(self.num_layers * self.num_directions,
                                batch_size, self.hidden_dim)
            h0.data.uniform_(-init, init)
            c0.data.uniform_(-init, init)
            hx = [h0, c0]
            hx = [x.to(device) for x in hx]
            if self.rnn_type in ['RNN', 'GRU', 'MyGRU']:
                hx = hx[0]
        return hx

    def init_weights(self, padding_idx=None):
        init = 0.1
        if self.embed_input is not None:
            self.embed_input.weight.data.uniform_(-init, init)

        if padding_idx:
            if self.embed_input is not None:
                self.embed_input.weight.data[padding_idx] = 0


# def get_full_io_masks_batch(seq_events):
#     """
# seq_events: (n_batch x n_timesteps x n_events)
# output : 
#     - full_mask : n_batch x [[n_timesteps x n_events]] x [[n_timesteps x n_events]]
#     - io_mask (input-output) : n_batch n_batch x [[n_timesteps x n_events]] x n_events
#     """

#     full_mask

def subseq_nth_mask(nth_steps_batch, max_steps, padding_value=0):
    """
    !! assume nth_steps value starts from 1  

    - test case: 
        subseq_nth_mask([torch.Tensor([1,1,1,2,2,2,3]), torch.Tensor([1,2,3,3,3])])
    - input nth_steps_batch : (n_batch x element)
    - output batch mask matrix : 
        - full mask: (n_batch x max_seq_len x max_seq_len)
        - io (input-output) mask : (n_batch x max_seq_len x max_seq_len)

    - in mask, row represents output dimension and col represents input dim
    
    Example (single instance):
    - input:
    [1,1,2,2]
    
    - full mask: 
    1 1 
    1 1
    1 1 1 1
    1 1 1 1

    - io mask:
    1 1 
    1 1 1 1

    each mask is started from upper-left. when it is smaller than largest, 
    padding value will be filled.
    """    
    device = nth_steps_batch[0].device
    n_batch = len(nth_steps_batch)
    max_seq_len = max([x.size(0) for x in nth_steps_batch])
    full_mask_batch = torch.ones(n_batch, max_seq_len, max_seq_len) * padding_value
    io_mask_batch = torch.ones(n_batch, max_steps, max_seq_len) * padding_value
    
    for i, nth_steps in enumerate(nth_steps_batch):
        # assume 1d tensor (i.e., batch-specific-element)
        assert nth_steps.dim() == 1
        length = nth_steps.size(0)
        full_mask = torch.stack([nth_steps] * length)
        io_mask = torch.stack([nth_steps] * max_steps)
        full_mask_batch[i, :length, :length] = full_mask
        io_mask_batch[i, :max_steps, :length] = io_mask
    
    full_mask_batch = full_mask_batch.to(device)
    io_mask_batch = io_mask_batch.to(device)

    padding_full_mask = full_mask_batch != 0
    padding_io_mask = io_mask_batch != 0
    full_mask_batch = (full_mask_batch <= torch.transpose(full_mask_batch, 1,2)) * padding_full_mask
    _range = torch.Tensor([list(range(max_steps))] * max_seq_len).to(device)\
        .t().expand(n_batch, max_steps, max_seq_len) + 1
    io_mask_batch = (io_mask_batch <= _range) * padding_io_mask
    return full_mask_batch.bool(), io_mask_batch.bool()
