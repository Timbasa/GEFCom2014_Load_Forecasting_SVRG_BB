from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_layer = 1
hidden_layer = 50
number_layer = 2
output_layer = 1
quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
daysperweek = 7
hoursperday = 24


# RNN, many to one, lstm
class RNN(nn.Module):
    def __init__(self, input_layer, hidden_layer, number_layer, output_layer):
        super(RNN. self).__init__()
        self.hidden_layer = hidden_layer
        self.num_layer = number_layer
        self.lstm = nn.LSTM(input_layer, hidden_layer, number_layer, batch_first=True)
        self.fc = nn.Linear(hidden_layer, output_layer)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Quantile_loss_function(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i,q in enumerate(quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Quantile_loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def pre_train():
    print("reading file")
    task1_train = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 1/L1-train.csv', header=None)
    #the times that have load value start from 35066 to end
    task1_train_array = task1_train.values[35065:]
    task1_train_time = task1_train_array[:, 1]
    task1_train_load = task1_train_array[:, 2]
    task1_train_station = task1_train_array[:, np.arange(3, 28)]

    hour = (int)(task1_train_load.shape[0] / hoursperday)
    X_train = np.reshape(task1_train_load, (hour, hoursperday, 1))
    X_train = np.array(X_train)
    mean_load = X_train.mean()
    X_train -= mean_load
    model = RNN(input_layer, hidden_layer, number_layer, output_layer)
