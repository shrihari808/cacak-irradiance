"""Mixed recurrent models: an LSTM layer followed by one or two GRU layers."""

import torch.nn as nn

from common import Head, Recurrent, run_sequence_models


class Net(nn.Module):
    def __init__(self, hidden_sizes, cells, in_features=2):
        super().__init__()
        self.rnn = Recurrent(in_features, hidden_sizes, cells)
        self.head = Head(self.rnn.out_features)

    def forward(self, x):
        return self.head(self.rnn(x))


MODELS = {
    "lstm_gru_1": lambda: Net([64, 32], ["lstm", "gru"]),
    "lstm_gru_2": lambda: Net([64, 32, 16], ["lstm", "gru", "gru"]),
}

if __name__ == "__main__":
    run_sequence_models("LSTM-GRU", MODELS)
