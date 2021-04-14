# Logistic Regression
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

        # xavier initialization
        torch.nn.init.xavier_normal(self.linear.weight)
        self.linear.bias.data.fill_(-5)

    def forward(self, x):
        out = self.linear(x)
        return out
