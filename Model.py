import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, drop):
        super(RNN,self).__init__()
        self.drop = drop
        self.encoder = nn.Embedding()