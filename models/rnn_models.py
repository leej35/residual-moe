# BNLSTM, LSTMCell and LSTM class from https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
# GRU is made based on jihunchoi's implementation


import torch
from torch.autograd import Variable
from torch.nn import functional, init
import torch.nn as nn

debug=False

class SeparatedBatchNorm1d(nn.Module):
    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, timestep):
        self._check_input_dim(input_)
        if timestep >= self.max_length:
            timestep = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(timestep))
        running_var = getattr(self, 'running_var_{}'.format(timestep))
        return functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class BNLSTMCell(nn.Module):
    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(BNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        # BN parameters
        self.bn_ih = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(
            num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        # The input-to-hidden weight matrix is initialized orthogonally.
        init.orthogonal(self.weight_ih.data)
        # The hidden-to-hidden weight matrix is initialized as an identity
        # matrix.
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        init.constant(self.bias.data, val=0)
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, timestep):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            timestep: The current timestep value, which is used to
                get appropriate running statistics.
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, timestep=timestep)
        bn_wi = self.bn_ih(wi, timestep=timestep)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
                                 split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, timestep=timestep))
        return h_1, c_1


class LSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
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
        """
        Initialize parameters following the way proposed in the paper.
        """

        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        if len(h_0.size()) > 2:
            h_0 = h_0.squeeze()
            c_0 = c_0.squeeze()
        batch_size = input_.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        try:
            wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        except:
            raise

        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                             split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
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
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))  # returning self.cell_1

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        _truncate_max_len = True  # instead of running all size of length of sequence,
        # get the max of mini batch sequence length as max timestep
        # I checked both results are same. (True/False)
        if _truncate_max_len:
            max_time = torch.max(length).item()
        else:
            max_time = input_.size(0)
        output = []
        for timestep in range(max_time):
            if isinstance(cell, BNLSTMCell):
                h_next, c_next = cell(input_=input_[timestep], hx=hx, timestep=timestep)
            else:
                h_next, c_next = cell(input_=input_[timestep], hx=hx)
            if debug: print('h_next:{}\ntime:{}\nlength:{}'.format(h_next, timestep, length))
            mask = (timestep < length).float().unsqueeze(1).expand_as(h_next)

            if input_.is_cuda:
                device = input_.get_device()
                mask = mask.to(device)

            # if debug: print('mask:{}'.format(mask))
            h_next = mask * h_next + (1 - mask) * hx[0]
            c_next = mask * c_next + (1 - mask) * hx[1]
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx=None, length=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.to(device)
        elif isinstance(length, list):
            length = Variable(torch.LongTensor(length))
            if input_.is_cuda:
                length = length.cuda()
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx)
            # Not the correct way of doing dropout?
            # if self.dropout > 0:
            #     input_ = self.dropout_layer(layer_output)
            # else:
            #     input_ = layer_output
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)




class GRUCell(nn.Module):
    """A basic GRU cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        """
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def reset_parameters(self):
        self.w_z = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.u_z = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.b_z = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.w_r = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.u_r = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.b_r = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.w_h = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.u_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx   : A (batch, hidden_size) tensor contains the initial hidden
                state.

        Returns:
            h_1: Tensor containing the next hidden state.

        Reference:
            https://en.wikipedia.org/wiki/Gated_recurrent_unit
        """

        # h_0, c_0 = hx
        h_0 = hx
        batch_size = h_0.size(0)

        z = self.w_z(input_) + self.u_z(h_0) + self.b_z.unsqueeze(0).expand(batch_size, *self.b_z.size())
        z = torch.sigmoid(z)

        r = self.w_r(input_) + self.u_r(h_0) + self.b_r.unsqueeze(0).expand(batch_size, *self.b_r.size())
        r = torch.sigmoid(r)

        h = self.w_h(input_) + self.u_h(r * h_0) + self.b_h.unsqueeze(0).expand(batch_size, *self.b_h.size())
        h = torch.tanh(h)

        h_1 = z * h + (1 - z) * h_0

        return h_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class GRU(nn.Module):
    """A module that runs multiple steps of GRU."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(GRU, self).__init__()
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
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))  # returning self.cell_1

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        _truncate_max_len = True  # instead of running all size of length of sequence,
        # get the max of mini batch sequence length as max timestep
        # I checked both results are same. (True/False)
        if _truncate_max_len:
            max_time = torch.max(length).data[0]
        else:
            max_time = input_.size(0)
        output = []
        for timestep in range(max_time):
            h_next = cell(input_=input_[timestep], hx=hx)
            if debug: print('h_next:{}\ntime:{}\nlength:{}'.format(h_next, timestep, length))
            mask = (timestep < length).float().unsqueeze(1).expand_as(h_next)
            if debug: print('mask:{}'.format(mask))
            hx_next = mask * h_next + (1 - mask) * hx
            output.append(hx_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx=None, length=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size), requires_grad=False)
        elif isinstance(length, list):
            length = Variable(torch.LongTensor(length), requires_grad=False)
        if input_.is_cuda:
            device = input_.get_device()
            length = length.to(device)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
        h_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, layer_h_n = self._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx)
            # Not the correct way of doing dropout?
            # if self.dropout > 0:
            #     input_ = self.dropout_layer(layer_output)
            # else:
            #     input_ = layer_output
            h_n.append(layer_h_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        return output, h_n




def main():
    pass

if __name__ == '__main__':
    main()

