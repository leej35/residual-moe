import torch.nn as nn
from models.layer_lstm import LayerLSTM, LayerLSTMCell

class ResidualSeqNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, input_dim, num_layers, dropout, device, rnn_type='MyLayerLSTM'):
        super(ResidualSeqNet, self).__init__()
        self.rnn_type = rnn_type
        self.embed_input = nn.Linear(input_dim, embed_dim, bias=False)
        if rnn_type == 'MyLayerLSTM':
            self.rnn = LayerLSTM(
                cell_class=LayerLSTMCell,
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                device=device)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=num_layers,
                dropout=dropout)
            self.rnn = self.rnn.to(device)

        else:
            print("rnn_type is not defined")
            raise NotImplementedError

        self.fc_out_residual = nn.Linear(hidden_dim, input_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq_events, lengths, hidden, trg_times=None, inp_times=None):

        input_seq = self.embed_input(seq_events)

        if self.rnn_type == 'GRU':
            _output, hidden = self.rnn(input_seq, hidden)
        else:
            _output, hidden = self.rnn(input_seq, hidden, lengths)
            _output = _output.transpose(0, 1)

        plain_output = self.fc_out(_output) + self.fc_out_residual(_output)

        return plain_output, None, hidden
