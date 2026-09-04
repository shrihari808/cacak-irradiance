"""Dilated causal front end feeding one or two GRU layers.

The front end extracts local features across the window; the recurrent layers
model their order. Variant 1 uses a single layer of 32 units, variant 2 stacks
32 and 16.
"""

import torch.nn as nn

from common import TCNFront, Head, Recurrent, run_sequence_models


class Net(nn.Module):
    def __init__(self, hidden_sizes, in_features=2):
        super().__init__()
        self.front = TCNFront(in_features)
        self.rnn = Recurrent(self.front.out_features, hidden_sizes, "gru")
        self.head = Head(self.rnn.out_features)

    def forward(self, x):
        return self.head(self.rnn(self.front(x)))


MODELS = {
    "tcn_gru_1": lambda: Net([32]),
    "tcn_gru_2": lambda: Net([32, 16]),
}

if __name__ == "__main__":
    run_sequence_models("TCN-GRU", MODELS)
