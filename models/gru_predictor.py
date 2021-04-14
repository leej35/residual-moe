import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embed_dim, hidden_dim,
                          dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedded_inp):

        batch_size = embedded_inp.size(0)
        device = embedded_inp.device
        hidden = self.init_hidden(batch_size, self.hidden_dim).to(device)

        hidden, _ = self.gru(embedded_inp, hidden)

        prediction = self.proj_out(hidden)
        prediction = F.sigmoid(prediction)
        return prediction

    @staticmethod
    def init_hidden(batch_size, hidden_dim):
        init = 0.1
        h0 = torch.randn(batch_size, hidden_dim)
        h0.data.uniform_(-init, init)
        return h0.unsqueeze(0)
