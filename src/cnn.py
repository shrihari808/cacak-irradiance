"""1D convolutional models of increasing depth.

cnn_1 is a single valid convolution with no pooling; cnn_2 and cnn_3 use
same-padded convolutions with max-pooling between them. All three collapse the
time axis by global average pooling before the shared head.
"""

import torch.nn as nn

from common import CONV_KERNEL, Head, LEAKY_SLOPE, run_sequence_models


class CNNNet(nn.Module):
    def __init__(self, channels, in_features=2, pad_same=True, pool_after=()):
        super().__init__()
        padding = CONV_KERNEL // 2 if pad_same else 0
        layers = []
        prev = in_features
        for i, out_channels in enumerate(channels):
            layers += [
                nn.Conv1d(prev, out_channels, CONV_KERNEL, padding=padding),
                nn.LeakyReLU(LEAKY_SLOPE),
            ]
            if i in pool_after:
                layers.append(nn.MaxPool1d(2))
            prev = out_channels
        self.features = nn.Sequential(*layers)
        self.head = Head(prev)

    def forward(self, x):
        # Conv1d expects (batch, channels, time); the loaders yield (batch, time, features).
        x = self.features(x.permute(0, 2, 1))
        return self.head(x.mean(dim=2))


MODELS = {
    "cnn_1": lambda: CNNNet([64], pad_same=False),
    "cnn_2": lambda: CNNNet([64, 32], pool_after=(0,)),
    "cnn_3": lambda: CNNNet([64, 32, 16], pool_after=(0, 1)),
}

if __name__ == "__main__":
    run_sequence_models("CNN", MODELS)
