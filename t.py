import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
    
    def layer_maker(self, type, input_size, hidden_size, num_layers, output_size):
        if type == 'LSTM':
            lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'GRU':
            gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'RNN':
            rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'linear':
            lin = nn.Linear(input_size, hidden_size)

    def forward(self, x, struct: list[dict[str, dict[str, list | int]]], num, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        for layer in struct:
            for i in range(layer['sequence']['layer_num']):
                self.layer_maker(type=layer['sequence']['type'][i], input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

