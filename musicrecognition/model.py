import torch
from torch import nn

crit = nn.MSELoss()


class SimpleNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, fixed_length: int):
        super(SimpleNet, self).__init__()
        self.fixed_length = fixed_length

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=(1, 1))
        self.fc1 = nn.Linear(480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = x[:, :self.fixed_length, :].unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.6
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
