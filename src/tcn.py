"""Temporal convolutional models of increasing depth (64, 64-32, 64-32-16).

Each nominal layer is a block of TCN_LEVELS dilated causal convolutions at that
filter count, so the receptive field grows exponentially within a block. The
prediction is read from the last time step.
"""

import torch.nn as nn

from common import TCN, TCN_LEVELS, Head, run_sequence_models


class TCNNet(nn.Module):
    def __init__(self, filters, in_features=2):
        super().__init__()
        blocks = []
        prev = in_features
        for n_filters in filters:
            blocks.append(TCN(prev, [n_filters] * TCN_LEVELS))
            prev = n_filters
        self.blocks = nn.Sequential(*blocks)
        self.head = Head(prev)

    def forward(self, x):
        x = self.blocks(x.permute(0, 2, 1))
        return self.head(x[:, :, -1])


MODELS = {
    "tcn_1": lambda: TCNNet([64]),
    "tcn_2": lambda: TCNNet([64, 32]),
    "tcn_3": lambda: TCNNet([64, 32, 16]),
}

if __name__ == "__main__":
    run_sequence_models("TCN", MODELS)
