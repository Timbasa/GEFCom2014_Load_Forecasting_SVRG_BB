import torch
import pandas as pd
import numpy as np
from Data_Solver.reshape_data import reshape_data
from Model.lstm import LSTM


PATH = 'rnnmodel.pt'


def test_offlane(device, input_layer, hidden_layer, number_layer, output_layer):
    print('test offlane mode')
    rnn = RNN(input_layer, hidden_layer, number_layer, output_layer).to(device)
    rnn.load_state_dict(torch.load(PATH))
    rnn.eval()
    task1_train = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 1/L1-train.csv', header=None,
                              low_memory=False)
    pre_train = reshape_data(task1_train.values[1:], 0)
    predict = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 2/L2-train.csv', header=None,
                              low_memory=False)
    predict = reshape_data(predict.values[1:], 1)
    train = [pre_train[-number_layer:], pre_train[-number_layer:]]
    for i in range(0, len(predict), 24):
        train_tensor = [torch.tensor(train[0][-number_layer:]).view(1, number_layer, input_layer),
                        torch.tensor(train[1][-number_layer:]).view(1, number_layer, input_layer)]
        output_p50 = rnn(train_tensor[0])
        output_p90 = rnn(train_tensor[1])
        output_p50 = output_p50.view(-1, 2)
        output_p90 = output_p90.view(-1, 2)
        new_p50= output_p50[:, 0]
        new_p90 = output_p90[:, 1]
        for j in range(24):
            predict[i + j][0] = new_p50[j]
            train[0] = np.append(train[0], predict[i + j])
            train[0] = train[0].reshape(-1, input_layer)
            predict[i + j][0] = new_p90[j]
            train[1] = np.append(train[1], predict[i + j])
            train[1] = train[1].reshape(-1, input_layer)

    for i in range(len(train[0])):
        print('p50: ', train[0][i][0], 'p90: ', train[1][i][0])



if __name__ == '__main__':
    test_offlane()
