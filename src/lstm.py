"""LSTM models: one, two and three stacked layers (64, 64-32, 64-32-16)."""

import torch.nn as nn

from common import Head, Recurrent, run_sequence_models


class LSTMNet(nn.Module):
    def __init__(self, hidden_sizes, in_features=2):
        super().__init__()
        self.rnn = Recurrent(in_features, hidden_sizes, "lstm")
        self.head = Head(self.rnn.out_features)

    def forward(self, x):
        return self.head(self.rnn(x))


MODELS = {
    "lstm_1": lambda: LSTMNet([64]),
    "lstm_2": lambda: LSTMNet([64, 32]),
    "lstm_3": lambda: LSTMNet([64, 32, 16]),
}

if __name__ == "__main__":
    run_sequence_models("LSTM", MODELS)
