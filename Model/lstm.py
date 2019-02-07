import torch
import math
import numpy as np
import torch.nn as nn


# RNN, many to many, lstm
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, number_layer, output_size, output_layer):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.output_layer = output_layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=number_layer,
                            batch_first=True,
                            dropout=0.2)
        self.out = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(output_layer)])

    def forward(self, x):
        out, _ = self.lstm(x, None)
        out = torch.cat([layer(out[:, -1, :]) for layer in self.out], dim=1)
        out = out.view(out.size(0), self.output_size, self.output_layer)
        return out

    def train(self, device, train_data, validation, loss_function, optimizer, batch_size, epoch, train_loss, validation_loss):
        (x, y) = train_data
        (x_v, y_v) = validation
        for e in range(1, epoch + 1):
            len_batch = math.ceil(x.size(0) / batch_size)
            losses = []
            for batch_idx in range(len_batch):
                if batch_size * (batch_idx + 1) > x.size(0):
                    output = self.forward(x[batch_idx * batch_size:])
                    target = y[batch_idx * batch_size:]
                else:
                    output = self.forward(x[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                    target = y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                loss = loss_function(output, target)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e,
                                                                            batch_idx * batch_size, x.size(0),
                                                                            100. * batch_idx / len_batch, loss.item()))
            train_loss.append(np.mean(losses))
            losses.clear()
            pred = self.off_predict(x_v)
            los = loss_function(pred, y_v)
            validation_loss.append(los)
            print('Epoch:{} train loss:{}, validation loss:{}'.format(e, train_loss[e - 1], validation_loss[e - 1]))


    # pytorch lstm validate the result
    def off_predict(self, x):
        p = self.forward(x).view(-1, 1, self.output_layer)
        return p
