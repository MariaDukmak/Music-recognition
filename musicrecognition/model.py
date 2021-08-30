from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a
        )

    def forward(self):
        pass


class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, sequence, input_size]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, sequence, hidden_size]
        last_from_sequence = lstm_out[:, -1, :]
        # last_from_sequence: [batch, hidden_size]
        return self.lin(last_from_sequence)
        # return: [batch, output_size]
