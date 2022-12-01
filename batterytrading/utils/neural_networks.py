import torch.nn as nn
import torch


def get_lstm_model_pytorch(n_features, n_hidden=50, n_layers=2, dropout=0.5):
    """
    Returns a LSTM model in PyTorch.
    Args:
        n_features:
        n_past_values:
        n_future_values:
        n_hidden:
        n_layers:
        dropout:

    Returns:

    """

    model = nn.LSTM(
        input_size=n_features,
        hidden_size=n_hidden,
        num_layers=n_layers,
        dropout=dropout,
    )
    return model


class ManyToOneLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, fc_hidden=12, num_layers=2):
        super(ManyToOneLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, output_size),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Output dimension: (seq_length, batch_size,  hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :])
        return out


class Control_Net(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        output_size=1,
        fc_hidden=12,
        num_layers=2,
        action_dim=1,
    ):
        super(Control_Net, self).__init__()
        self.lstm = ManyToOneLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            fc_hidden=fc_hidden,
            num_layers=num_layers,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregator = nn.Sequential(
            nn.Linear(output_size + 1, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, action_dim),
        )

    def preprocess_input_unbatched(self, x):
        x_soc = x[0].to(self.device)
        x_price = x[1:].to(self.device)
        return x_soc, x_price

    def preprocess_input_batched(self, x):
        x_soc = x[:, 0].to(self.device)
        x_soc = x_soc.reshape(-1, 1)
        x_price = x[:, 1:].to(self.device)
        x_price = x_price.reshape(x_price.shape[1], x_price.shape[0], 1)
        return x_soc, x_price

    def forward(self, x_soc, x_price):
        out_lstm = self.lstm(x_price)
        interim_result = torch.hstack((out_lstm, x_soc))
        out = self.aggregator(interim_result)
        return out


def create_input_lstm(data):
    """Extract required entries from a data dictionary, stack them and return them as a numpy array"""
    soc_tensor = torch.tensor(data["SOC"].reshape((1, 1)), dtype=torch.float32)
    price_history_tensor = torch.tensor(data["historic_price"].reshape((-1, 1)), dtype=torch.float32)
    return torch.concat((soc_tensor, price_history_tensor), dim=0)


if __name__ == "__main__":
    from batterytrading.environment import Environment
    import numpy as np

    env = Environment()
    next_state, reward, done, info = env.step(0)
    action = np.array([0.0008])
    next_state, reward, done, info = env.step(action)
    model = Control_Net(input_size=1, hidden_size=99, output_size=4, fc_hidden=12, num_layers=2)
    x = create_input_lstm(next_state)
    x_soc, x_price = model.preprocess_input_unbatched(x)

    output = model(x_soc, x_price)
    print(output.shape)
