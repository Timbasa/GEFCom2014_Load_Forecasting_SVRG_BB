from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = F.relu()
        return F.log_softmax(x,dim=1)


def pre_train():
    print("reading file")
    task1_train = pd.read_csv('./GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 1/L1-train.csv', header=None)
    #the times that have load value start from 35066 to end
    task1_train_array = task1_train.values[35065:]
    task1_train_time = task1_train_array[:, 1]
    task1_train_load = task1_train_array[:, 2]
    task1_train_station = task1_train_array[:, np.arange(3, 28)]
