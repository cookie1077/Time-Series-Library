import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Model, self).__init__()
        self.rnn = torch.nn.GRU(self, hidden_dim, num_layers=layers, batch_first=True, dropout=0.5)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(hidden_dim, 20, bias=True)
        self.fc2 = torch.nn.Linear(20, 30, bias=True)
        self.fc3 = torch.nn.Linear(30, output_dim, bias=True)
        self.loss_function = torch.nn.MSELoss()

    def __init__(self, configs):
        super(Model, self).__init__()
        hidden_dim = 10
        layers = 3
        self.rnn = torch.nn.GRU(configs.enc_in, hidden_dim, num_layers=layers, batch_first=True, dropout=0.5)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(hidden_dim, 20, bias=True)
        self.fc2 = torch.nn.Linear(20, 30, bias=True)
        self.fc3 = torch.nn.Linear(30, configs.c_out, bias=True)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = x_enc
        x, _status = self.rnn(x)
        x = self.fc1(x[:,-1])
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        return self.loss_function(y, y_hat)