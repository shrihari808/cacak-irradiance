"""GRU models: one, two and three stacked layers (64, 64-32, 64-32-16)."""

import torch.nn as nn

from common import Head, Recurrent, run_sequence_models


class GRUNet(nn.Module):
    def __init__(self, hidden_sizes, in_features=2):
        super().__init__()
        self.rnn = Recurrent(in_features, hidden_sizes, "gru")
        self.head = Head(self.rnn.out_features)

    def forward(self, x):
        return self.head(self.rnn(x))


MODELS = {
    "gru_1": lambda: GRUNet([64]),
    "gru_2": lambda: GRUNet([64, 32]),
    "gru_3": lambda: GRUNet([64, 32, 16]),
}

if __name__ == "__main__":
    run_sequence_models("GRU", MODELS)
