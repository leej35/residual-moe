
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable
import torch.nn.init as init

from tabulate import tabulate
from .base_seq_model import (
    masked_bce_loss,
    sigmoid,
    relu, tanh, logsoftmax,
    fill_idx_multihot,
    bceloss, BaseMultiLabelLSTM)
from .periodic_predictor import PeriodicPredictor, AdaptivePeriodicPredictor
from .cluster_events import get_clusters
from .positional_embeddings import (
    init_weights,
    ReversePositionalEncoding,
    PositionalEncoding,
    BackPositionalEncoding,
    PositionalEncodingConcat)
from utils.tensor_utils import fit_input_to_output

import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

tabulate.PRESERVE_WHITESPACE = True
torch.manual_seed(5)

def masked_softmax(vector,
                   mask,
                   dim = -1,
                   memory_efficient= False,
                   mask_fill_value= -1e32):
    """
    From : https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L216

    ``torch.nn.functional.softmax(vector)`` does not work if some elements of
    ``vector`` should be masked.  This performs a softmax on just the non-masked
    portions of ``vector``.  Passing ``None`` in for the mask is also acceptable;
    you'll just get a regular softmax. ``vector`` can have an arbitrary number
    of dimensions; the only requirement is that ``mask`` is broadcastable to
    ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``,
    we will unsqueeze on dimension 1 until they match.

    If you need a different unsqueezing of your mask, do it yourself before
    passing the mask into this function. If ``memory_efficient`` is set to true,
    we will simply use a very large negative number for those masked positions
    so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is
    false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used
    as the last layer of a model that uses categorical cross-entropy loss.
    Instead, if ``memory_efficient``
    is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask,
            # we zero these out.
            result = F.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = F.softmax(masked_vector, dim=dim)
    return result

def get_event_group(event_dic):
    groups = {}
    for eid, info in event_dic.items():
        cate = info['category']
        if cate not in groups:
            groups[cate] = []
        groups[cate].append(eid)
    return groups


def _get_mask(tensor):
    use_cuda = tensor.is_cuda
    device = tensor.device

    # find nonzero vector
    n_b, n_e, n_d = tensor.size()
    z = torch.zeros(n_d)
    if use_cuda:
        z = z.to(device)
    # idxs = (.sum(2) != n_d).nonzero()
    return tensor != z


class AttnMultiLabelLSTM(BaseMultiLabelLSTM):
    def __init__(self, event_size, window_size_y, target_size, hidden_dim, embed_dim,
                 use_cuda=False, batch_size=64, batch_first=True,
                 bidirectional=False, out_func='sigmoid',
                 rnn_type='GRU', num_layers=1, 
                 dropout=0,
                 dropouth=0,
                 dropouti=0,
                 wdrop=0,
                 tie_weights=False,
                 padding_idx=0,
                 pred_time=False,
                 output_type='plain',
                 device=None,
                 is_input_time=False,
                 hier_pred=False,
                 recent_bias=False,
                 memorize_time=False,
                 is_pack_seq=False,
                 attn_type='dot',
                 attn_channel='event',
                 channel_size=4,
                 pe='None',
                 pec=False,
                 target_type='multi',
                 attn_q_type='None',
                 attn_direct=False,
                 attn_remove_zero_vectors=False,
                 attn_fast_masked_softmax=False,
                 event_dic=None,
                 train_dataset=None,
                 attn_group_type='event_type',
                 attn_temp_embed=False,
                 pooling_type='mean',
                 attn_co_channel=False,
                 train_data_path=None,
                 cluster_only=False,
                 cluster_method='discretize',
                 pred_period=False,
                 pp_type='adaptive',
                 pp_attn=False,
                 use_layer_norm=False,
                 pp_weight_scheme='default',
                 use_simple_gate=False,
                 use_orig_params=True,
                 init_decay=False,
                 group_size_method='fixed',
                 attn_inner_dim=None,
                 do_svd=False,
                 freeze_emb=False,
                 remap=False,
                 skip_hidden_state=False,
                 f_exp=False,
                 f_window=False,
                 rb_init="none",
                 manual_alpha=0,
                 off_lstm=False,
                 inv_id_mapping=None,
                 clock_gate=False,
                 pp_concat=False,
                 pp_merge_signal=None,
                 rb_concat=False,
                 pp_ascounts=False,
                 ):
        super(AttnMultiLabelLSTM, self).__init__(
            event_size=event_size, window_size_y=window_size_y,
            target_size=target_size, hidden_dim=hidden_dim, 
            embed_dim=embed_dim, use_cuda=use_cuda, batch_size=batch_size,
            batch_first=batch_first, bidirectional=bidirectional, out_func=out_func, 
            rnn_type=rnn_type, num_layers=num_layers, 
            padding_idx=padding_idx, pred_time=pred_time, 
            output_type=output_type, device=device, is_input_time=is_input_time,
            hier_pred=hier_pred, recent_bias=recent_bias, memorize_time=memorize_time,
            is_pack_seq=is_pack_seq, target_type=target_type,
            dropout=dropout, dropouti=dropouti, dropouth=dropouth, 
            wdrop=wdrop, tie_weights=tie_weights, pred_period=pred_period,
            pp_type=pp_type, use_simple_gate=use_simple_gate, 
            use_orig_params=use_orig_params, pp_weight_scheme=pp_weight_scheme,
            remap=remap, skip_hidden_state=skip_hidden_state,
            f_exp=f_exp,
            f_window=f_window,
            rb_init=rb_init,
            manual_alpha=manual_alpha,
            inv_id_mapping=inv_id_mapping,
            pp_concat=pp_concat,
            pp_merge_signal=pp_merge_signal,
            rb_concat=rb_concat,
            pp_ascounts=pp_ascounts,
            attn_channel=attn_channel,
            )
        # off_lstm=off_lstm,

        if rnn_type == 'NoRNN':
            hidden_dim = embed_dim
        if attn_inner_dim is None:
            attn_inner_dim = hidden_dim

        self.inv_id_mapping = inv_id_mapping
        self.hidden_dim = hidden_dim

        if not self.pred_period and self.rb_concat:
            self.fc_out = nn.Linear(hidden_dim, target_size)

            if tie_weights:
                self.fc_out.weight = self.embed_input.weight

            init.xavier_uniform_(self.fc_out.weight.data)
            init.constant_(self.fc_out.bias.data, 0)

        self.W1 = nn.Linear(hidden_dim, attn_inner_dim, bias=(not use_orig_params))
        self.W2 = nn.Linear(hidden_dim, attn_inner_dim, bias=(not use_orig_params))
        init.xavier_uniform_(self.W1.weight.data)
        init.xavier_uniform_(self.W2.weight.data)
        if not use_orig_params:
            init.constant_(self.W1.bias.data, 0)
            init.constant_(self.W2.bias.data, 0)

        self.softmax_d1 = nn.Softmax(dim=1)
        self.attn_type = attn_type # 'dot' or 'additive'
        self.attn_channel = attn_channel
        self.attn_q_type = attn_q_type
        self.attn_direct = attn_direct
        self.group_type = attn_group_type
        self.attn_temp_embed = attn_temp_embed
        self.pooling_type = pooling_type
        self.attn_co_channel = attn_co_channel
        self.use_layer_norm = use_layer_norm
        self.init_decay = init_decay
        self.use_orig_params = use_orig_params  # simpler method used for AIME

        self.pp_attn = pp_attn

        if pp_attn:
            self.W_pp_attn_ri = nn.Linear(target_size, hidden_dim)  # project recent interval
            self.W_pp_attn_ac = nn.Linear(target_size, hidden_dim)  # project accumulate counter
            self.W_pp_attn_ec = nn.Linear(
                target_size, hidden_dim)  # project elapsed counter
            self.W_pp_attn_combine = nn.Linear(hidden_dim, hidden_dim)

        if pooling_type == 'attn':
            self.attn_pool = nn.Linear(embed_dim, 1)

        if attn_temp_embed:
            self.attn_temp_embed_layer = TemporalAttnEmbedding(embed_dim)

        if attn_direct or pooling_type == 'max':
            self.embed_input = nn.Embedding(event_size, embed_dim,
                                            padding_idx=padding_idx)
            self.embed_input = init_weights(self.embed_input,
                                            padding_idx=padding_idx)
        self.recent_bias = recent_bias
        if recent_bias:
            self.r_bias_linear = nn.Linear(event_size, target_size)
            if not use_simple_gate:
                self.W_rb_gate_preds = nn.Linear(event_size, target_size)
                self.W_rb_gate_seqs = nn.Linear(event_size, target_size)

            if not self.use_orig_params:
                self.W_rb_gate = nn.Linear(target_size * 2, target_size)

        if attn_channel == 'single':
            if attn_q_type == 'None':
                self.q = nn.Parameter(torch.FloatTensor(1, hidden_dim))
            elif attn_q_type == 'recent':
                self.q = nn.Linear(event_size, hidden_dim)
        elif attn_channel in ['group', 'event']:
            self.channel_size = event_size if attn_channel == 'event' else channel_size
            self.q = nn.Parameter(torch.FloatTensor(self.channel_size, hidden_dim))

        if attn_q_type != 'None' and attn_q_type == 'attn_q_type':
            self.q = init_weights(self.q)

        if attn_channel == 'group':

            if self.group_type == 'event_type':
                self.groups = get_event_group(event_dic)

            elif self.group_type == 'cooccur':

                assert train_dataset is not None

                if cluster_only:
                    for _assign_labels in ['discretize']:
                        for _channel_size in [8, 16, 32]:
                            print("\n\n\n\n\n\n")
                            print("=" * 50)
                            print("assign_labels:{}   n cluster:  {}"
                                  "".format(_assign_labels, _channel_size))
                            print("=" * 50)
                            self.groups, embeding_svd = get_clusters(event_size,
                                                        cluster_size=_channel_size,
                                                        train_data=train_dataset,
                                                        batch_size=64,
                                                        target_type=self.target_type,
                                                        d_path=train_data_path,
                                                        event_dic=event_dic,
                                                        assign_labels=_assign_labels,
                                                        method=cluster_method,
                                                        verbose=True,
                                                        embed_dim=embed_dim,
                                                        do_svd=do_svd)
                    exit()
                else:
                    print("\n\n\n\n\n\n")
                    print("=" * 50)
                    print("assign_labels:{}   n cluster:  {}".format(
                        cluster_method, channel_size))
                    print("=" * 50)
                    self.groups, embeding_svd = get_clusters(event_size,
                                                cluster_size=channel_size,
                                                train_data=train_dataset,
                                                batch_size=64,
                                                target_type=self.target_type,
                                                d_path=train_data_path,
                                                event_dic=event_dic,
                                                method=cluster_method,
                                                verbose=True,
                                                embed_dim=embed_dim,
                                                do_svd=do_svd)

                # fetch svd-based embedding
                if do_svd:
                    if str(self.embed_input).startswith('Linear'):
                        self.embed_input.weight.data = embeding_svd.t().to(self.device)
                    else:
                        embedding_svd = None
                        self.embed_input.from_pretrained(
                            embeddings=embedding_svd.to(self.device),
                            padding_idx=padding_idx,
                            freeze=freeze_emb)

            self.g_attn = GroupAttentionInterface(
                self.groups, hidden_dim, event_size, dropout,
                 attn_type=attn_type,
                 attn_q_type=attn_q_type,
                 use_cuda=use_cuda, device=self.device,
                 fast_masked_softmax=attn_fast_masked_softmax,
                 loss_combine=True,
                 use_co_channel=self.attn_co_channel,
                 group_size_method=group_size_method)
            self.fc = nn.Linear(channel_size * hidden_dim, event_size)

        self.use_pe = pe
        self.use_pec = pec
        # self.use_rpe = attn_reverse_pe

        if self.use_pe in ['sinusoid', 'pos']:
            self.pe = PositionalEncoding(d_model=embed_dim, dropout=dropout,
                                     max_len=1000, sparse_sinusoid=self.attn_direct,
                                     use_sinusoid=(self.use_pe == 'sinusoid'))
        elif self.use_pe in ['reverse-pos']:
            self.pe = ReversePositionalEncoding(embed_dim, dropout, max_len=1000,
                                                attn_direct=self.attn_direct,
                                                init_decay=self.init_decay)
        elif self.use_pe in ['back']:
            self.pe = BackPositionalEncoding(embed_dim, dropout, max_len=1000)
        elif self.use_pe in ['concat']:
            self.pe = PositionalEncodingConcat(d_model=embed_dim, d_posemb=2000,
                                                dropout=dropout, max_len=1000)


        if use_layer_norm:
            self.norm_layer = nn.LayerNorm(hidden_dim)

        self.target_type = target_type

        self.remove_zero_vectors = attn_remove_zero_vectors
        self.fast_masked_softmax = attn_fast_masked_softmax
        self.event_dic = event_dic

    def forward(self, seq_events, lengths, hidden, trg_times=None, inp_times=None, method='sparse'):

        if self.target_type == 'multi':
            seq_events = seq_events.float()

        if self.attn_direct:
            seq_events_idx = fill_idx_multihot(seq_events, method=method)
            if method == 'sparse':
                seq_events_idx, seq_time_idx = seq_events_idx
            else:
                seq_time_idx = None

            input_seq = self.embed_input(seq_events_idx)

        else:
            if self.pooling_type == 'mean':
                input_seq = self.embed_input(seq_events)
                
                if self.target_type == 'single':
                    assert seq_events.sum(0).sum(0)[0] == 0, \
                        "!!first dim event is not zero"
                seq_time_idx = None

            elif self.pooling_type in ['max', 'attn']:

                sparse_mode = False

                if sparse_mode:
                    seq_events_idx, seq_time_idx \
                        = fill_idx_multihot(seq_events, method='sparse')

                    max_step = seq_time_idx.max()

                    step_masks = [step == seq_time_idx for step in range(max_step + 1)]
                    masked_inps = [mask.long() * seq_events_idx for mask in step_masks]

                    if self.pooling_type == 'max':
                        emb_bags = [self.embed_input(inps).max(1)[0] for inps in masked_inps]
                        pooled_emb = torch.stack(emb_bags)
                        input_seq = pooled_emb.permute(1,0,2)
                    elif self.pooling_type == 'attn':
                        emb_bags = []
                        for inps in masked_inps:
                            emb = self.embed_input(inps)
                            alpha = F.tanh(self.attn_pool(emb))
                            alpha = F.softmax(alpha)
                            pooled_emb = alpha * emb

                    else:
                        raise NotImplementedError

                    # when size not match with original, add zero padding after
                    while input_seq.size(1) < seq_events.size(1):
                        z_pad = torch.zeros(input_seq.size(0),
                                            1, input_seq.size(2)).to(self.device)
                        input_seq = torch.cat((input_seq, z_pad), dim=1)

                else:
                    light_mode = False
                    if light_mode:
                        seq_events_idx = fill_idx_multihot(seq_events,
                                                                 method='default')
                        seq_time_idx = None
                        input_seq = self.embed_input(seq_events_idx)
                        input_seq, _ = torch.max(input_seq, dim=2)
                    else:
                        n_b, n_s, n_e = seq_events.size()
                        # TODO: FIX contiguous
                        _tmp_input = seq_events.contiguous().view(-1, n_e).long()
                        input_seq = self.embed_input(_tmp_input)
                        input_seq = input_seq.contiguous().view(n_b, n_s, n_e, -1)
                        input_seq = input_seq.sum(2)

            else:
                raise NotImplementedError

        # if self.attn_temp_embed:
            # self.attn_temp_embed_layer = self.attn_temp_embed_layer(input_seq)

        if self.use_pe in ['sinusoid', 'pos', 'reverse-pos']:
            input_seq = self.pe(input_seq, time_seq=seq_time_idx,
                                lengths=lengths)
        elif self.use_pe in ['None']:
            pass
        else:
            raise NotImplementedError

        if self.rnn_type in ['MyLSTM']:
            hidden_steps, (h_n, c_n) = self.rnn(input_seq, hidden, lengths)
            hidden = (h_n, c_n)
            hidden_steps = self.dropout_layer(hidden_steps)

        elif self.rnn_type in self.hier_lstms:
            hidden_steps, hidden = self.rnn(input_seq, hidden, lengths)
            hidden_steps = self.dropout_layer(hidden_steps)

        if self.rnn_type == 'NoRNN':
            hidden_steps = input_seq

        if self.rnn_type in self.hier_lstms + ['MyLSTM'] and self.batch_first:
            hidden_steps = hidden_steps.transpose(0, 1)

        """NOTE:
        hidden_steps: n_batch x max_seq_len x hidden_dim
        """
        preds = []
        max_time_step = seq_events.size(1)

        if self.pred_period:
            self.pp.initialize_batch(
                hidden_steps.size(0), 
                self.event_size if not self.lab_pp_proc else self.target_size,
                device=self.device)

        pp_steps = []
        pp_out = None
        for t in range(1, max_time_step + 1):

            if self.pred_period:
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

            pred_step = self.pred_step(hidden_steps[:, :t, :],
                                       seq_events[:,t-1,:],
                                       periodic_signal=pp_out)

            preds.append(pred_step)

        if self.attn_channel == 'event':
            preds = torch.stack(preds,dim=1)
        else:
            preds = torch.cat(preds, dim=1)
        
        if self.pred_period:
            pp_preds = torch.stack(pp_steps, dim=1).to(self.device)

        if self.use_layer_norm:
            preds = self.norm_layer(preds)

        if self.attn_channel in ['single'] and not self.pred_period:
            preds = self.fc_out(preds)

        if self.recent_bias and not self.pred_period:
            r_bias = self.r_bias_linear(seq_events.float())
            if preds.dim() > r_bias.dim():
                preds = preds.squeeze(-2)
            preds = preds + r_bias

        if self.pred_period:


            # AIM-Journal version
            preds = torch.cat((preds, pp_preds), dim=2)
            self.fc_out.weight.data.mul_(self.pp.event_mask)
            preds = self.fc_out(preds)
            self.pp.pp.deinitialize()  # free up ephemeral parameters

            if self.recent_bias:
                rb_signal = self.r_bias_linear(seq_events.float())
                preds = preds + rb_signal


            # periodic_pred = torch.stack(pp_steps, dim=1)
            # if self.inv_id_mapping is not None:
            #     periodic_pred = fit_input_to_output(
            #         periodic_pred, self.inv_id_mapping.keys())

            # if self.use_cuda:
            #     periodic_pred = periodic_pred.to(self.device)

            # if self.use_orig_params:
            #     if preds.dim() == 4 and preds.size(2) == 1:
            #         preds = preds.squeeze(2)
            #     preds = preds + self.pp_weight * periodic_pred
            # else:
            #     pp_gate = self.pp_gate(preds, periodic_pred)
            #     preds = preds + F.sigmoid(pp_gate) * periodic_pred


            # self.pp.pp.deinitialize()  # free up ephemeral parameters

        assert (preds != preds).sum() == 0, "NaN!"

        return preds, None, hidden

    def rb_gate(self, preds, seq_events):
        if self.use_simple_gate:
            w_seqs = seq_events
            w_preds = preds
        else:
            w_preds = F.tanh(self.W_rb_gate_preds(preds))
            w_seqs = F.tanh(self.W_rb_gate_seqs(seq_events))
        w_concat = torch.cat((w_preds, w_seqs), dim=2)
        gate = self.W_rb_gate(w_concat)
        return gate


    def pred_step(self, _hidden_steps, recent_events=None, periodic_signal=None):

        if self.attn_channel == 'single':

            W = self.W1(_hidden_steps)

            # prepare query

            if self.attn_q_type == 'None':
                q = self.W2(self.q)
            elif self.attn_q_type == 'recent':
                assert recent_events is not None
                q = self.W2(self.q(recent_events))

                while len(q.size()) < len(W.size()):
                    q = q.unsqueeze(1)

                if q.size() != W.size():
                    q = q.expand_as(W)
            else:
                raise NotImplementedError

            if self.pp_attn:
                assert periodic_signal is not None

                w_ac = F.tanh(self.W_pp_attn_ac(self.pp.pp.ac))
                w_ec = F.tanh(self.W_pp_attn_ec(self.pp.pp.ec))
                w_ri = F.tanh(self.W_pp_attn_ri(self.pp.pp.ri))

                w_pp_attn = F.sigmoid(self.W_pp_attn_combine(w_ac + w_ec + w_ri))
                while len(W.size()) > len(w_pp_attn.size()):
                    w_pp_attn = w_pp_attn.unsqueeze(1)

            if self.attn_type == 'additive':

                if self.pp_attn:
                    weights = sigmoid(W + q + w_pp_attn)
                else:
                    weights = sigmoid(W + q)

            elif self.attn_type == 'dot':
                if self.pp_attn:
                    weights = W * q * w_pp_attn
                else:
                    weights = W * q

            if self.fast_masked_softmax:
                _mask = _get_mask(_hidden_steps)
                weights = masked_softmax(weights, _mask , dim=1,
                                         memory_efficient=True)
            else:
                weights = self.softmax_d1(weights)

            s = weights * _hidden_steps
            pred = torch.sum(s, dim=1)

        elif self.attn_channel == 'group':
            return self.g_attn(_hidden_steps, recent_events)

        elif self.attn_channel == 'event':

            n_batch = _hidden_steps.size(0)
            seq_len = _hidden_steps.size(1)

            q = self.W2(self.q)
            x = self.W1(_hidden_steps)

            # if seq_len > 1: 
            #     x = x.unsqueeze(2)

            x = x.expand(n_batch, seq_len, self.channel_size, self.hidden_dim)

            if self.attn_type == 'additive':
                weights = sigmoid(x + q)

            elif self.attn_type == 'dot':
                weights = x * q

            weights = self.softmax_d1(weights)
            """ 
            weights : n_batch x seqlen x n_events x hidden_dim
            """
            s = weights * _hidden_steps

            if self.attn_channel == 'event':
                s = torch.sum(s, dim=1) # sum over timesteps
                pred = torch.sum(s, dim=2) # sum over hidden dims == predict by event
            elif self.attn_channel == 'group':
                # let's concatenate channels and project to event dimension
                s = s.view(s.size(0), s.size(1), -1)
                s = self.fc(s)
                pred = torch.sum(s, dim=1) # sum over timesteps

        return pred



class GroupAttentionInterface(nn.Module):
    def __init__(self, groups, hidden_dim, event_size, dropout,
                 attn_type='additive',
                 attn_q_type='None',
                 use_cuda=False,
                 device=None,
                 fast_masked_softmax=False,
                 loss_combine=True,
                 use_co_channel=False,
                 group_size_method='fixed'
                 ):
        super(GroupAttentionInterface, self).__init__()
        self.groups = groups  # dict, key: group_idx, value: class_idxs
        self.loss_combine = loss_combine
        self.event_size = event_size
        self.hidden_dim= hidden_dim
        self.group_size_method = group_size_method

        if group_size_method.startswith('renorm'):
            scale = float(group_size_method.split('renorm')[-1])
        else:
            scale = None

        idims = self.get_group_dims(groups, scale)

        print('idims:', idims)

        self.nn_modules = nn.ModuleDict({group: GroupAttnModule(
            hidden_dim, len(itemids), event_size, dropout, attn_type, attn_q_type,
            fast_masked_softmax, internal_dim=idims[group])
            for group, itemids in groups.items()})

        self.group_idxs = {}
        for group, itemids in self.groups.items():
            _z_idx = torch.zeros(1, len(itemids))

            if use_cuda:
                _z_idx = _z_idx.to(device)

            for _in_idx, _out_idx in enumerate(itemids):
                _z_idx[0, _in_idx] = _out_idx
            self.group_idxs[group] = _z_idx
        if use_co_channel:
            self.co_channel = GroupAttnModule(hidden_dim, event_size, event_size,
                                              dropout, attn_type, attn_q_type,
                                              fast_masked_softmax)
        self.use_co_channel = use_co_channel


    def get_group_dims(self, groups, scale=None):
        if self.group_size_method == 'hidden':
            scheme = {k: self.hidden_dim for k, v in groups.items()}
        elif self.group_size_method.startswith('fixed'):
            try:
                val = int(self.group_size_method.split('fixed')[-1])
            except ValueError:
                val = 1
            scheme = {k: val for k, v in groups.items()}
        else:
            denom = np.sum([len(x) for x in list(groups.values())])
            ng = {k: len(v) / denom for k, v in groups.items()}
            scaled_ng = {k: v ** scale for k, v in ng.items()}
            denom2 = np.sum([x for x in list(scaled_ng.values())])
            scheme = {k: int(self.hidden_dim * v / denom2)
                      for k, v in scaled_ng.items()}
        return scheme


    def forward(self, _hidden_steps, recent_events=None):
        device = _hidden_steps.device
        pred_groups = {group: self.nn_modules[group](_hidden_steps, recent_events) \
                for group in list(self.groups.keys())}

        n_b, n_hists, hidden_dim = _hidden_steps.size()

        if self.loss_combine:
            pred = Variable(torch.zeros(n_b, self.event_size))

            pred = pred.to(device)

            for group in list(self.groups.keys()):
                g_pred = pred_groups[group]
                g_idx = self.group_idxs[group].expand_as(g_pred).long()

                pred.scatter_(1, g_idx.to(device), g_pred.to(device))
        else:
            raise NotImplementedError

        if self.use_co_channel:
            pred += self.co_channel(_hidden_steps, recent_events)
        return pred


class GroupAttnModule(nn.Module):
    def __init__(self, hidden_dim, output_size, event_size, dropout,
                 attn_type='additive',
                 attn_q_type='None',
                 fast_masked_softmax=False,
                 internal_dim=None):
        super(GroupAttnModule, self).__init__()
        self.fast_masked_softmax = fast_masked_softmax

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        i_dim = hidden_dim if internal_dim is None else internal_dim

        self.fc_out = nn.Linear(hidden_dim, output_size)
        # init.xavier_uniform_(self.fc_out.weight.data)
        # init.constant_(self.fc_out.bias.data, 0)

        self.W1 = nn.Linear(hidden_dim, i_dim, bias=False)
        self.W2 = nn.Linear(i_dim, i_dim, bias=False)
        # init.xavier_uniform_(self.W1.weight.data)
        # init.xavier_uniform_(self.W2.weight.data)

        self.q = nn.Linear(event_size, i_dim)
        if not fast_masked_softmax:
            self.softmax_d1 = nn.Softmax(dim=1)
        self.attn_q_type = attn_q_type
        self.attn_type = attn_type

    def forward(self, _hidden_steps, recent_events=None):

        W = self.W1(_hidden_steps)

        if self.attn_q_type == 'None':
            q = self.W2(self.q)
        elif self.attn_q_type == 'recent':
            assert recent_events is not None
            q = self.W2(self.q(recent_events))

            while len(q.size()) < len(W.size()):
                q = q.unsqueeze(1)

            if q.size() != W.size():
                q = q.expand_as(W)
        else:
            raise NotImplementedError

        if self.attn_type == 'additive':
            weights = sigmoid(W + q)

        elif self.attn_type == 'dot':
            weights = W * q
        else:
            raise NotImplementedError

        if self.fast_masked_softmax:
            _mask = _get_mask(_hidden_steps)
            weights = masked_softmax(weights, _mask, dim=1,
                                     memory_efficient=True)
        else:
            weights = self.softmax_d1(weights)

        s = weights * _hidden_steps
        pred = torch.sum(s, dim=1)
        pred = self.fc_out(pred)
        return pred


class TemporalAttnEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TemporalAttnEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_kernel = None


    def _create_temporal_embedding(self, seq_len):
        n_batch, max_seq_len = len(seq_len), max(seq_len)

        batch_kernels = []
        for b_len in seq_len:
            """
            _beta: controls number of fluctuation in a full seq length
            1 : monotonic decreasing (increasing)
            2 : single decrease and then increase
            4 : repeat _beta=2 two times
            8 : ... 
            """
            kernels = []
            for _beta in [1, 2, 4, 8, 16]:
                x = (torch.cos(torch.arange(0, b_len).float()\
                               .div(b_len/_beta/math.pi)) + 1) * 0.5
                x_inv = (torch.cos(torch.arange(0, b_len).float()\
                                   .add(b_len*math.pi)\
                                   .div(b_len/_beta/math.pi)) + 1) * 0.5

                if max_seq_len - b_len > 0:
                    # add zero padding
                    _pad = torch.zeros(max_seq_len - b_len)
                    x = torch.cat([x, _pad])
                    x_inv = torch.cat([x_inv, _pad])

                kernels.append(x)
                kernels.append(x_inv)
            kernels = torch.stack(kernels)
            batch_kernels.append(kernels)

        batch_kernels = torch.stack(batch_kernels)

        return batch_kernels


    def forward(self, x, seq_len):
        """

        :param x: n_batch x max_seq_len x embed_dim [zero padded]
        :param seq_len: n_batch; length of each element sequence
        :return: n_batch x max_seq_len x embed_dim
        """

        temporal_kernels = self._create_temporal_embedding(seq_len)
        _x = x * temporal_kernels
