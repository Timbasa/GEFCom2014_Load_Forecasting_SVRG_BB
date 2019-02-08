from __future__ import print_function

import os
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from Data_Solver.reshape_data import reshape_data
from Data_Solver.to_supervised import to_surpervised
from Data_Solver.quantile_loss import QuantileLoss
from Data_Solver.data_loader import DataLoader
from Model.lstm import LSTM
import matplotlib.pyplot as plt

PATH = 'rnnmodel.pt'
quantiles = [0.5, 0.9]
input_size = 20
hidden_size = 100
number_layer = 1
batch_size = 32
epoch = 50
input_layer = 48
output_layer = len(quantiles)
output_size = 24
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loss = []
validation_loss = []
task1_train_start = 35064
# task1_train_start = 76680

def plot_results(prediction_list, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i in range(len(prediction_list)):
        plt.plot(prediction_list[i], label='Prediction')
    plt.legend()
    plt.show()


def pre_train():
    print("reading file")
    configs = json.load(open('config.json', 'r'))
    # task1_train = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 1/L1-train.csv', header=None, low_memory=False)
    data = DataLoader(
        os.path.join(configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        task1_train_start
    )
    scaler = MinMaxScaler()
    scaler.fit(data.data_train)
    loss_function = QuantileLoss(quantiles)
    train_data = reshape_data(scaler.transform(data.data_train), 0)
    x, y = to_surpervised(train_data, input_layer, output_size, 'train')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    validation_data = reshape_data(scaler.transform(data.data_test), 1)
    x_validation, y_validation = to_surpervised(validation_data, input_layer, output_size, 'validation')
    x_validation = torch.tensor(x_validation, dtype=torch.float32).to(device)
    y_validation = torch.tensor(y_validation, dtype=torch.float32).to(device)
    y_validation = y_validation.view(-1, 1)
    # test = task1_train.values[35065:]
    # scaler.fit(np.reshape(reshapeddata[:, 0], (-1, 1)))
    # reshapeddata = reshape_data(task1_train.values[35065:], 0)
    # origin_data = reshapeddata
    # scaler_data = scaler.transform()
    # x_train, y_train = to_surpervised(reshapeddata[-8760:-2190], input_layer, output_size, 'train')
    # x_validation, y_validation = to_surpervised(reshapeddata[-2190:], input_layer, output_size, 'validation')
    # x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    # y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    # x_validation = torch.tensor(x_validation, dtype=torch.float32).to(device)
    # y_validation = torch.tensor(y_validation, dtype=torch.float32).to(device)
    # y_validation = y_validation.view(-1, 1)
    model = LSTM(input_size, hidden_size, number_layer, output_size, output_layer).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.2)
    model.train(device, (x, y), (x_validation, y_validation), loss_function, optimizer, batch_size, epoch, train_loss,
                validation_loss, scaler)
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # torch.save(model.state_dict(), PATH)
    # test_offlane(device, input_layer, hidden_layer, number_layer, output_layer)
    predictions = model.off_predict(x_validation)
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    prediction_list = []
    for i in range(output_layer):
        prediction_list.append(scaler.inverse_transform(predictions[:, :, i].cpu().detach().numpy()))
    prediction_inv = scaler.inverse_transform(predictions.view(-1, 1).cpu().detach().numpy())
    y_validation_inv = scaler.inverse_transform(y_validation.cpu().detach().numpy())
    plot_results(prediction_list, y_validation_inv)
    print("quantile loss: {}".format(loss_function(torch.tensor(prediction_inv).view(-1, 1, output_layer), torch.tensor(y_validation_inv))))

if __name__ == '__main__':
    pre_train()
