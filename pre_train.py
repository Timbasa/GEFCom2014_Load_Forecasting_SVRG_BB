from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from loader import Loader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from reshape_data import reshape_data
from to_supervised import to_surpervised
from quantile_loss import QuantileLossFunction
from model import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = 'rnnmodel.pt'

quantiles = [0.5, 0.99]
input_layer = 20
hidden_layer = input_layer * 4
number_layer = 48
output_layer = len(quantiles)


def train(model, device, train_loader, test_loader, optimizer, epoch):
    loss_func = QuantileLossFunction(quantiles).to(device)
    for e in range(1, epoch+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.view(output.shape[0], -1, 2)
            # print(output[0])
            # print(target[0])
            # print(output[1])
            # output = output.reshape(output.shape[0], 2, 24)
            output = output.to(device)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        #validation
        sum_loss = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(output.shape[0], -1, 2)
            print('Q50: ', output[0, :, 0])
            print('Q90: ', output[0, :, 1])
            print(target[0])
            loss = loss_func(output, target)
            sum_loss.append(loss.item())
        print('Train Epoch: {} , the test Loss is {:.6f}'.format(e, np.mean(sum_loss)))


def pre_train():
    print("reading file")
    task1_train = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 1/L1-train.csv', header=None, low_memory=False)
    #the times that have load value start from 35066 to end
    reshapeData = reshape_data(task1_train.values[35065:], 0)
    x_train, y_train = to_surpervised(reshapeData[-8760:-6576])
    x_test, y_test = to_surpervised(reshapeData[-6576:])
    train_loader = torch.utils.data.DataLoader(Loader(x_train, y_train), batch_size=64, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(Loader(x_test, y_test), batch_size=64, shuffle=True, pin_memory=True)
    rnnmodel = RNN(input_layer, hidden_layer, number_layer, output_layer).to(device)

    optimizer = optim.SGD(rnnmodel.parameters(), lr= 0.3, momentum=0.2)

    train(rnnmodel, device, train_loader, test_loader, optimizer, epoch=200)

    torch.save(rnnmodel.state_dict(), PATH)

    print('test offlane mode')
    # rnn = RNN(input_layer, hidden_layer, number_layer, output_layer).to(device)
    # rnn.load_state_dict(torch.load(PATH))
    # rnn.eval()
    task1_train = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 1/L1-train.csv', header=None,
                              low_memory=False)
    pre_train = reshape_data(task1_train.values[1:], 0)
    predict = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 2/L2-train.csv', header=None,
                              low_memory=False)
    predict = reshape_data(predict.values[1:], 1)
    per_train = [pre_train[-48:], pre_train[-48:]]
    for i in range(0, len(predict), 24):
        train_tensor = [torch.tensor(per_train[0][-48:]).view(1, 48, input_layer).to(device),
                        torch.tensor(per_train[1][-48:]).view(1, 48, input_layer).to(device)]
        output_p50 = rnnmodel(train_tensor[0])
        output_p90 = rnnmodel(train_tensor[1])
        output_p50 = output_p50.view(-1, 2)
        output_p90 = output_p90.view(-1, 2)
        new_p50= output_p50[:, 0]
        new_p90 = output_p90[:, 1]
        for j in range(24):
            predict[i + j][0] = new_p50[j]
            per_train[0] = np.append(per_train[0], predict[i + j])
            per_train[0] = per_train[0].reshape(-1, input_layer)
            predict[i + j][0] = new_p90[j]
            per_train[1] = np.append(per_train[1], predict[i + j])
            per_train[1] = per_train[1].reshape(-1, input_layer)

    for i in range(len(per_train[0])):
        print('p50: ', per_train[0][i][0], 'p90: ', per_train[1][i][0])
#
if __name__ == '__main__':
    pre_train()