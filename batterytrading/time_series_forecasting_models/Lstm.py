import numpy as np
import torch
import torch.nn as nn
from torch import optim


class LstmEncoder(nn.Module):
    """Encodes time-series sequence"""

    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        """

        super(LstmEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x_input):
        """
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        """

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """

        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


class LstmDecoder(nn.Module):
    """Decodes hidden state output by encoder"""

    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        """
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        """

        super(LstmDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        # self.decoder = nn.Linear(hidden_size, output_size)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, output_size))
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        """
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        """

        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        # output = self.linear(lstm_out.squeeze(0))
        output = self.decoder(lstm_out)
        return output, self.hidden


if __name__ == "__main__":
    from batterytrading.data_loader import Data_Loader_batch

    data_loader = Data_Loader_batch(16, price_time_horizon=7, share_of_samples_start=0.0, share_of_samples_end=0.7)
    encoder = LstmEncoder(1, hidden_size=64, num_layers=3)

    decoder = LstmDecoder(input_size=64, hidden_size=64, output_size=1, num_layers=3)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_function = nn.MSELoss()
    y_ls = []
    output_ls = []
    loss_ls = []
    for i in range(5000):
        # while data_loader.has_next:
        x, y = next(data_loader)
        encoded, h = encoder(x)
        # encoded = encoded[:, -1, :]
        out, _ = decoder(encoded, h)
        loss = loss_function(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        y = y.detach().numpy()
        out = out.detach().numpy()
        y_ls.append(y)
        output_ls.append(out)
        loss_ls.append(loss.detach().numpy())
        if i % 100 == 0:
            running_loss = np.mean(loss_ls[-100:])
            print(f"Iteration {i}, loss: {loss}, running_loss {running_loss}")

    import matplotlib.pyplot as plt

    y_ls = np.stack(y_ls)
    output_ls = np.stack(output_ls)
    s = y_ls.shape

    y_ls = y_ls.reshape(s[0] * s[1], s[2], s[3])
    output_ls = output_ls.reshape(s[0] * s[1], s[2], s[3])
    np.save("y.npy", y_ls)
    np.save("output.npy", output_ls)
    plt.plot(y_ls[:, -1, 0])
    plt.plot(output_ls[:, -1, 0])
    plt.show()
    plt.plot(y_ls[:, 0, 1])
    plt.plot(output_ls[:, 0, 1])
    plt.show()
