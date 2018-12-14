import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# RNN, many to many, lstm
class RNN(nn.Module):
    def __init__(self, input_layer, hidden_layer, number_layer, output_layer, output_per_layer= 1):
        super(RNN, self).__init__()
        self.hidden_layer = hidden_layer
        self.num_layer = number_layer
        self.lstm = nn.LSTM(input_layer, hidden_layer, number_layer, batch_first=True)
        self.fc = nn.Linear(hidden_layer, output_layer)
        # self.fc = nn.ModuleList([nn.Linear(self.hidden_layer, output_per_layer) for _ in range(output_layer)])

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        x = x.float()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out = torch.cat([layer(out[:, -1, :]) for layer in self.fc], dim=1)
        return out