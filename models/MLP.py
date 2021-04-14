# Logistic Regression
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim):
        super(MLP, self).__init__()
        self.linear_input = nn.Linear(input_size, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(hidden_dim, num_classes)

        # xavier initialization
        torch.nn.init.xavier_normal(self.linear_input.weight)
        self.linear_input.bias.data.fill_(1e-5)
        torch.nn.init.xavier_normal(self.linear1.weight)
        self.linear1.bias.data.fill_(1e-5)
        torch.nn.init.xavier_normal(self.linear_output.weight)
        self.linear_output.bias.data.fill_(1e-5)

    def forward(self, x):
        out = F.tanh(self.linear_input(x))
        out = F.tanh(self.linear1(out))
        out = self.linear_output(out)
        return out
