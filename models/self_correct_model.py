import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence, pad_sequence
)
from .straight_through_estimator import StraightThroughEstimator
from .base_seq_model import masked_bce_loss

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)



class PredictorLSTM(nn.Module):
    def __init__(
        self,
        rnn_class,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        dropout=0,
        num_layers=1,
        bidirectional=False,
    ):
        super(PredictorLSTM, self).__init__()
        
        self.proj_inp = nn.Linear(
            input_dim, embed_dim, bias=False
        )
        self.lstm = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout
        )
        self.proj_out = nn.Linear(
            hidden_dim, output_dim
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inp, hidden, lengths):
        embedding = self.proj_inp(inp)

        _embedding = pack_padded_sequence(
            embedding, lengths, batch_first=True)

        hidden, _ = self.lstm(_embedding, hidden)

        hidden, _ = pad_packed_sequence(
            hidden,
            batch_first=True,
            total_length=max(lengths)
        )

        hidden = self.dropout_layer(hidden)
        prediction = torch.sigmoid(self.proj_out(hidden))
        return prediction, hidden, embedding


class CorrectorLSTM(nn.Module):
    def __init__(
        self,
        rnn_class,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0,
        num_layers=1,
        bidirectional=False,
        activation='tanh',
        init_bias_zero=False,
        init_weight_small=False,
        init_val=0.0001,
    ):
        super(CorrectorLSTM, self).__init__()

        self.lstm = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout
        )
        self.proj_out = nn.Linear(
            hidden_dim, output_dim, 
        )
        
        # initialize bias of projection as zero
        if init_bias_zero:
            self.proj_out.bias.data.zero_()
        if init_weight_small:
            self.proj_out.weight.data.normal_(-init_val, init_val)

        self.activation = activation

        if activation == 'tanh':
            self.f_act = torch.tanh
        elif activation == 'relu':
            self.f_act = torch.relu
        elif activation == 'leakyrelu':
            self.f_act = F.leaky_relu

    def forward(self, inp, hidden, lengths):
        inp = pack_padded_sequence(
            inp, lengths, batch_first=True)

        hidden, _ = self.lstm(inp, hidden)

        hidden, output_lengths = pad_packed_sequence(
            hidden,
            batch_first=True,
            total_length=max(lengths)
        )
        if self.activation == 'none':
            correct_amount = self.proj_out(hidden)
        else:
            correct_amount = self.f_act(self.proj_out(hidden))

        return correct_amount, hidden


class CorrectorController(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embed_dim,
        target_dim,
        corrector_control_act,
        corrector_control_inp,
        corrector_control_arch,
    ):
        super(CorrectorController, self).__init__()

        self.corrector_control_act = corrector_control_act
        self.corrector_control_inp = corrector_control_inp
    
        
        if self.corrector_control_inp == "hidden_and_input":
            input_dim = embed_dim + hidden_dim
        elif self.corrector_control_inp == "hidden":
            input_dim = hidden_dim
        elif self.corrector_control_inp == "input":
            input_dim = embed_dim
        else:
            raise NotImplementedError
        
        if self.corrector_control_act == "sigmoid":
            self.f_act_out = F.sigmoid
        elif self.corrector_control_act == "hard":
            self.f_act_out = StraightThroughEstimator()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, target_dim)
        )

    def forward(
        self,
        inp,
    ):
        if type(inp) == tuple:
            inp = torch.cat(inp, dim=2)

        return self.f_act_out(self.layers(inp))




class SelfCorrectLSTM(nn.Module):
    def __init__(self,              
                 event_input_dim,
                 target_dim,
                 embed_dim,
                 hidden_dim,
                 rnn_type,
                 use_cuda=False,
                 device=None,
                 bidirectional=False,
                 num_layers=1,
                 dropout=0,
                 feed_input=False,
                 feed_input_and_hidden=False,
                 correct_loss_type='bce',
                 activation='tanh',
                 init_bias_zero=False,
                 init_weight_small=False,
                 init_val=0.0001,
                 no_clamp=False,
                 use_corrector_control=False,
                 corrector_control_act="None",
                 corrector_control_inp="hidden",
                 corrector_control_arch="mlp",
    ):
        super(SelfCorrectLSTM, self).__init__()
        
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.feed_input = feed_input
        self.feed_input_and_hidden = feed_input_and_hidden
        self.use_cuda = use_cuda
        self.device = device
        self.no_clamp = no_clamp
        self.use_corrector_control = use_corrector_control
        self.corrector_control_inp = corrector_control_inp

        self.use_mse = True if correct_loss_type == 'mse' else False

        corrector_input_dim = hidden_dim

        if self.feed_input:
            corrector_input_dim = embed_dim
        elif self.feed_input_and_hidden:
            corrector_input_dim += embed_dim

        if rnn_type == 'GRU':
            rnn_class = nn.GRU
        elif rnn_type == 'LSTM':
            rnn_class = nn.LSTM
        else:
            raise NotImplementedError

        # assume predictor model is pretrained.
        self.predictor = PredictorLSTM(
            rnn_class,
            event_input_dim,
            embed_dim,
            hidden_dim,
            target_dim,
            dropout,
            num_layers,
            bidirectional
        ).to(self.device)

        self.corrector = CorrectorLSTM(
            rnn_class,
            corrector_input_dim,
            hidden_dim,
            target_dim,
            dropout,
            num_layers,
            bidirectional,
            activation,
            init_bias_zero,
            init_weight_small,
            init_val
        ).to(self.device)

        if use_corrector_control:
            self.corrector_controller = CorrectorController(
                hidden_dim, 
                embed_dim,
                target_dim,
                corrector_control_act,
                corrector_control_inp,
                corrector_control_arch
            ).to(self.device)

    def forward(self, input_seq, input_len, trg_seq, hidden, run_mode):

        hidden_predictor, hidden_corrector = hidden, hidden.clone()

        if run_mode == 'train_corrector':
            # freeze weights
            self.predictor.proj_inp.weight.requires_grad = False
            self.predictor.proj_out.weight.requires_grad = False
            self.predictor.proj_out.bias.requires_grad = False

            for param in self.predictor.lstm.parameters():
                param.requires_grad = False

        pred, hidden_predictor, input_emb = self.predictor(
            input_seq, hidden_predictor, input_len
        )

        if run_mode != 'train_corrector':
            loss = masked_bce_loss(pred, trg_seq.float(), input_len)
            return loss, pred
                
        if self.feed_input:
            corrector_inp = input_emb
        elif self.feed_input_and_hidden:
            corrector_inp = torch.cat((hidden_predictor, input_emb), dim=2)
        else:
            corrector_inp = hidden_predictor

        correct_amount, hidden_corrector = self.corrector(
            corrector_inp, hidden_corrector, input_len
        )

        if self.use_corrector_control:
            if self.corrector_control_inp == "hidden_and_input":
                cc_inp = (input_emb, hidden_predictor)
            elif self.corrector_control_inp == "hidden":
                cc_inp = hidden_predictor
            elif self.corrector_control_inp == "input":
                cc_inp = input_emb
            else:
                raise NotImplementedError

            cc_v = self.corrector_controller(cc_inp)
            correct_amount = correct_amount * cc_v

        pred = pred + correct_amount
        
        if not self.no_clamp:
            pred = pred.clamp(0, 1)

        loss = masked_bce_loss(pred, trg_seq.float(), input_len,
                               use_mse=self.use_mse)

        return loss, pred

    def init_hidden(self, batch_size=None, device=None):
        init = 0.1
        if device is None:
            device = self.device

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
