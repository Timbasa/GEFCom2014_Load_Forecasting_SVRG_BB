import torch
import pandas as pd
import numpy as np
from reshape_data import reshape_data
from model import RNN


PATH = 'rnnmodel.pt'
quantiles = [0.5, 0.99]
input_layer = 44
hidden_layer = input_layer * 4
number_layer = 2
output_layer = len(quantiles)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prediction(model, data):
    output = []
    loss = []


def test_offlane():
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
    train = [pre_train[-48:], pre_train[-48:]]
    for i in range(len(predict)):
        train_tensor = [torch.tensor(train[0][-48:]).view(1, 48, 44), torch.tensor(train[1][-48:]).view(1, 48, 44)]
        new_p50= rnn(train_tensor[0])[-1][0]
        new_p90 = rnn(train_tensor[1])[-1][1]
        predict[i][0] = new_p50
        train[0] = np.append(train[0], predict[i])
        train[0] = train[0].reshape(-1, 44)
        predict[i][0] = new_p90
        train[1] = np.append(train[1], predict[i])
        train[1] = train[1].reshape(-1, 44)

    print('done')

if __name__ == '__main__':
    test_offlane()
