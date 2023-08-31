import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# function to create train, test data given stock data and sequence length
def load_data(df, look_back):
    data_raw = df.values # convert to numpy array
    target_raw = df.pop('close').values
    data, target = [], []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])
        target.append(target_raw[index: index + look_back])

    data = np.array(data)
    target = np.array(target)
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = target[:train_set_size,:-1]

    x_test = data[train_set_size:,:-1]
    y_test = target[train_set_size:,:-1]

    return [x_train, y_train, x_test, y_test]

def make_tensor(df):
    look_back = 60 # choose sequence length
    x_train, y_train, x_test, y_test = load_data(df, look_back)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    return x_train, x_test, y_train, y_test

class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]
