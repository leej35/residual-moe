# BNLSTM, LSTMCell and LSTM class
# from https://github.com/jihunchoi/
# recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
# GRU is made based on jihunchoi's implementation





import torch
from torch.autograd import Variable
from torch.nn import functional as F, init
import torch.nn as nn


class LayerLSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LayerLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)

        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LayerLSTM(nn.Module):
    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, device=None):
        super(LayerLSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.device = device
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, event_seq, hx=None, length=None, time_seq=None):
        """
        :param event_seq:
        :param hx:
        :param length:
        :param time_seq:
        :return output: n_batch x maxseqlen x num_layers x hidden_dim
        :return hx: n_layer x (h_next, c_next)
        """
        if self.batch_first:
            try:
                event_seq = event_seq.transpose(0, 1)
            except:
                print(event_seq.size())
            if time_seq:
                time_seq = time_seq.transpose(0,1)
        max_time, batch_size, _ = event_seq.size()
        if isinstance(length, list):
            length = torch.LongTensor(length)
            if event_seq.is_cuda:
                length = length.to(self.device)
        output = [[] for x in range(self.num_layers)]
        for timestep in range(max_time):
            for layer in range(self.num_layers):
                cell = self.get_cell(layer)

                if layer == 0:
                    cell_input = event_seq[timestep]
                else:
                    cell_input = output[layer - 1][timestep]
                    cell_input = self.dropout_layer(cell_input)

                # cell_input: bptt x event_size

                h_next, c_next = cell(input_=cell_input, hx=hx[layer])

                mask = (timestep < length).float().unsqueeze(1).expand_as(
                    h_next)
                if event_seq.is_cuda:
                    mask = mask.to(self.device)

                h_next = mask * h_next + (1 - mask) * hx[layer][0]
                c_next = mask * c_next + (1 - mask) * hx[layer][1]

                output[layer].append(h_next)
                hx[layer] = (h_next, c_next)

        output = [torch.stack(l, 0) for l in output]
        output = torch.stack(output, 2)
        return output, hx


class GatedFeedbackLSTM(nn.Module):
    """
    https://github.com/hehaodele/GatedFeedback-LSTM.pytorch/blob/master/GFLSTM.py
    """
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0):
        super(GatedFeedbackLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        self.l_i2h = [nn.Linear(input_size, hidden_size * 3)]
        self.l_h2h = [nn.Linear(hidden_size, hidden_size * 3)]
        self.l_wc = [nn.Linear(input_size, hidden_size)]
        for L in range(1, num_layers):
            self.l_i2h.append(nn.Linear(hidden_size, hidden_size * 3))
            self.l_h2h.append(nn.Linear(hidden_size, hidden_size * 3))
            self.l_wc.append(nn.Linear(hidden_size, hidden_size))
        
        self.l_wg = []
        self.l_ug = []
        self.l_uc = []
        for L in range(num_layers):
            _wg, _ug, _uc = [], [], []
            for _L in range(num_layers):
                if L == 0:
                    _wg.append(nn.Linear(input_size, hidden_size))
                else:
                    _wg.append(nn.Linear(hidden_size, hidden_size))
                _ug.append(nn.Linear(hidden_size * num_layers, hidden_size))
                _uc.append(nn.Linear(hidden_size, hidden_size))
            self.l_wg.append(_wg)
            self.l_ug.append(_ug)
            self.l_uc.append(_uc)

        # set attributes
        for L in range(num_layers):
            setattr(self, 'layer_i2h_%d' % L, self.l_i2h[L])
            setattr(self, 'layer_h2h_%d' % L, self.l_h2h[L])
            setattr(self, 'layer_wc_%d' % L, self.l_wc[L])
        for L in range(num_layers):
            for _L in range(num_layers):
                setattr(self, 'layer_wg_%d_%d' % (L,_L), self.l_wg[L][_L])
                setattr(self, 'layer_ug_%d_%d' % (L,_L), self.l_ug[L][_L])
                setattr(self, 'layer_uc_%d_%d' % (L,_L), self.l_uc[L][_L])
        self.l_drop = nn.Dropout(dropout, inplace=True)

    def forward_one_step(self, input, hidden):
        nowh, nowc = hidden  # num_layers x batch_size x hidden_size
        nowH = nowh.transpose(0, 1).contiguous().view(nowh.size(1), -1)  # concate all hidden states
        nxth_list, nxtc_list = [], []

        for L in range(self.num_layers):
            if L > 0:
                input = self.l_drop(nxth_list[L - 1])  # (batch, input_size / hidden_size)
            h, c = nowh[L], nowc[L]  # (batch, hidden_size)
            i2h, h2h = self.l_i2h[L](input), self.l_h2h[L](h)  # (batch, hidden_size * 3)
            # cell gates
            i_gate, f_gate, o_gate = torch.split(F.sigmoid(i2h + h2h),
                                                   self.hidden_size,
                                                   dim=1)  # (batch, hidden_size)
            # global gates
            global_gates = []
            for _L in range(self.num_layers):
                global_gates.append(F.sigmoid(self.l_wg[L][_L](input) + self.l_ug[L][_L](nowH)))
            
            # decode in transform
            in_from_input = self.l_wc[L](input)
            for _L in range(self.num_layers):
                in_from_nowh = global_gates[_L] * self.l_uc[L][_L](nowh[_L])
                in_from_input = in_from_input + in_from_nowh
            in_from_input = F.tanh(in_from_input)
            
            # update cells and hidden
            _c = f_gate * c + i_gate * in_from_input
            _h = o_gate * F.tanh(_c)
            nxth_list.append(_h)
            nxtc_list.append(_c)

        # (num_layers, batch, hidden_size)
        nxth = torch.stack(nxth_list, dim=0)
        nxtc = torch.stack(nxtc_list, dim=0)
        output = nxth_list[-1]  # top hidden is output
        return output, (nxth, nxtc)

    def forward(self, input, hidden, length):
        if self.batch_first: # seq_first to batch_first
            input = input.transpose(0, 1)

        output = []
        for _in in input:
            _out, hidden = self.forward_one_step(_in, hidden)
            output.append(_out)
        output = torch.stack(output, dim=0)
        
        # if self.batch_first:
        #     output = output.transpose(0, 1)
        return output, hidden



class HierSparseTimeLSTM(LayerLSTM):
    def __init__(self, *args, **kwargs):
        super(HierSparseTimeLSTM, self).__init__(*args,**kwargs)
        self.time_weight = nn.Linear(self.input_size, 1)
        self.time_weight.bias.data.normal_(mean=0, std=1)

    def forward(self, event_seq, hx=None, length=None, time_seq=None):

        if self.batch_first:
            event_seq = event_seq.transpose(0, 1)
            if time_seq:
                time_seq = time_seq.transpose(0, 1)
        max_time, batch_size, _ = event_seq.size()
        if isinstance(length, list):
            length = torch.LongTensor(length)
            if event_seq.is_cuda:
                length = length.to(self.device)
        output = [[] for x in range(self.num_layers)]
        for timestep in range(max_time):
            for layer in range(self.num_layers):
                cell = self.get_cell(layer)

                if layer == 0:
                    cell_input = event_seq[timestep]
                else:
                    cell_input = output[layer - 1][timestep]
                    cell_input = self.dropout_layer(cell_input)

                h_next, c_next = cell(input_=cell_input, hx=hx[layer])

                if layer > 0 and time_seq:
                    threshold = F.relu(self.time_weight(time_seq))
                    skip_mask = (time_seq[timestep] > threshold).float().\
                        unsqueeze(1).expand_as(h_next)  # 0: skip the dim
                    h_next = skip_mask * h_next + (1 - skip_mask) * hx[layer][0]
                    c_next = skip_mask * c_next + (1 - skip_mask) * hx[layer][1]

                # copy prev hidden for zero padded elements
                mask = (timestep < length).float().unsqueeze(1).expand_as(
                    h_next)
                h_next = mask * h_next + (1 - mask) * hx[layer][0]
                c_next = mask * c_next + (1 - mask) * hx[layer][1]

                output[layer].append(h_next)
                hx[layer] = (h_next, c_next)

        output = [torch.stack(l, 0) for l in output]
        output = torch.stack(output, 2)
        return output, hx


class HierRateOfChangeLSTM(LayerLSTM):
    def __init__(self, *args, **kwargs):
        super(HierRateOfChangeLSTM, self).__init__(*args,**kwargs)
        self.memory = []
        self.k = 10  # moving average/variance window size

    def forward(self, event_seq, hx=None, length=None, time_seq=None):

        if self.batch_first:
            event_seq = event_seq.transpose(0, 1)
            if time_seq:
                time_seq = time_seq.transpose(0, 1)
        max_time, batch_size, _ = event_seq.size()
        if isinstance(length, list):
            length = torch.LongTensor(length)
            if event_seq.is_cuda:
                length = length.to(self.device)
        output = [[] for x in range(self.num_layers)]
        for timestep in range(max_time):
            for layer in range(self.num_layers):
                cell = self.get_cell(layer)

                if layer == 0:
                    cell_input = event_seq[timestep]
                else:
                    cell_input = output[layer - 1][timestep]
                    cell_input = self.dropout_layer(cell_input)

                h_next, c_next = cell(input_=cell_input, hx=hx[layer])

                if layer > 0:
                    start = max(len(output[layer])-self.k, 1)
                    hist = torch.stack(output[layer][start:])
                    if event_seq.is_cuda:
                        hist = hist.to(self.device)
                    rate_of_change = torch.var(hist) / len(hist.size(0))

                    threshold = F.relu(self.time_weight(time_seq))
                    skip_mask = (time_seq[timestep] > threshold).float().\
                        unsqueeze(1).expand_as(h_next)
                    h_next = skip_mask * h_next + (1 - skip_mask) * hx[layer][0]
                    c_next = skip_mask * c_next + (1 - skip_mask) * hx[layer][1]

                # copy prev hidden for zero padded elements
                mask = (timestep < length).float().unsqueeze(1).expand_as(
                    h_next)
                h_next = mask * h_next + (1 - mask) * hx[layer][0]
                c_next = mask * c_next + (1 - mask) * hx[layer][1]

                output[layer].append(h_next)
                hx[layer] = (h_next, c_next)

        output = [torch.stack(l, 0) for l in output]
        output = torch.stack(output, 2)
        return output, hx


def test_lstm():
    embed_dim = 2
    hidden_dim = 3
    cell_class = LayerLSTMCell
    embed = torch.nn.Embedding(10, embed_dim, padding_idx=0)
    torch_lstm = LayerLSTM(cell_class, embed_dim, hidden_dim, batch_first=True,
                           num_layers=2)

    input = [list(range(10)),list(range(5))+[0]*5,list(range(3))+[0]*7,list(range(10))]
    a_batch = torch.LongTensor(input)
    input = embed(a_batch)
    h_seq, h_last = torch_lstm(input)
    # h_seq:  n_batch x max_seq_len x n_layers x hidden_dim
    # h_last: (n_batch x n_layers x hidden_dim), (n_batch x n_layers x hidden_dim)
    print(h_seq)
    print(h_seq.size())


def main():
    test_lstm()
    # pass

if __name__ == '__main__':
    main()

