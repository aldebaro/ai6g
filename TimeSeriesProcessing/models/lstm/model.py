import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np


class LSTM1(nn.Module):
    def __init__(self, encoder_vector_sz=1, output_vector_sz=1, lstm_units_per_layer=100, num_layers=1, dp_percent=0.1, device='cpu'):
        super(LSTM1, self).__init__()
        self.num_classes = output_vector_sz  # number of output classes
        self.num_layers = num_layers  # number of layers
        self.input_size = encoder_vector_sz  # input size
        self.hidden_size = lstm_units_per_layer  # hidden state -> number of LSTM units
        self.device = device

        self.lstm = nn.LSTM(input_size=encoder_vector_sz,
                            hidden_size=lstm_units_per_layer,
                            num_layers=1,
                            batch_first=True,
                            dropout=dp_percent)  # lstm layer
        if self.num_layers == 2:
            self.lstm1 = nn.LSTM(input_size=lstm_units_per_layer,
                                 hidden_size=lstm_units_per_layer,
                                 num_layers=1,
                                 batch_first=True,
                                 dropout=dp_percent)  # second lstm layer
        self.fc_1 = nn.Linear(lstm_units_per_layer,
                              self.hidden_size)  # fully connected 1

        self.fc_2 = nn.Linear(lstm_units_per_layer,
                              self.hidden_size)  # fully connected 1

        self.fc = nn.Linear(self.hidden_size,
                            output_vector_sz)  # fully connected last layer

        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dp_percent)

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(
            self.device)  # hidden state
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(
            self.device)  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0))  # lstm with input, hidden, and internal state

        if self.num_layers == 2:
            # If we use a stacked LSTM architecture, we need to pass all hidden
            # states of the last layer to the new lstm layer, this way we get the
            # output variable as a entry to this second layer
            h_1 = Variable(torch.zeros(1,
                                       output.size(0), self.hidden_size)).to(
                                           self.device)  # hidden state
            c_1 = Variable(torch.zeros(1,
                                       output.size(0), self.hidden_size)).to(
                                           self.device)  # internal state
            output = self.dropout(output)
            output, (hn, cn) = self.lstm1(output, (h_1, c_1))

        # hn = hn.view(
        #     -1, self.hidden_size)  # hn contain the last hidden state output
        # # Note that we get the last hidden information to pass forward to the MLP layers,
        # # this is standard procedure for LSTM layers processing
        # First Dense
        out = self.dropout(output)
        out = self.fc_1(out)
        out = self.act_func(out)
        out = self.dropout(out)

        out = self.fc_2(out)
        out = self.act_func(out)
        out = self.dropout(out)
        # Last layer
        out = self.fc(out)
        #out = self.act_func(out)
        return out

    def predict(self, input_n):

        return self(input_n).cpu().detach().numpy()[0]


def main():
    device = torch.device('cpu')
    print('Using device:', device)
    model = LSTM1(device=device,
                  output_vector_sz=2,
                  encoder_vector_sz=1).to(device)
    x = torch.rand(60, 120, 1).to(device)

    out = model(x)
    print("out")


if __name__ == '__main__':
    main()
