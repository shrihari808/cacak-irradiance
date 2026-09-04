"""Feed-forward ANN baselines with one, two and three hidden layers.

The window is flattened and passed through fully connected ReLU layers, so the
model has no explicit notion of temporal order and serves as the conventional
neural baseline against which the recurrent, convolutional and graph
architectures are compared. Hidden widths follow the same 64, 64-32, 64-32-16
progression used by the recurrent families.
"""

import torch.nn as nn

from common import Head, run_sequence_models, SEQ_LEN


class ANNNet(nn.Module):
    def __init__(self, hidden_sizes, in_features=2):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_features * SEQ_LEN
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        self.body = nn.Sequential(*layers)
        self.head = Head(prev)

    def forward(self, x):
        return self.head(self.body(x))


MODELS = {
    "ann_1": lambda: ANNNet([64]),
    "ann_2": lambda: ANNNet([64, 32]),
    "ann_3": lambda: ANNNet([64, 32, 16]),
}

if __name__ == "__main__":
    run_sequence_models("ANN", MODELS)
