"""Graph model over the months of each input window.

A 12-month window is treated as a path graph whose nodes are the individual
months, each linked to its immediate neighbours in both directions. A two-level
temporal convolution produces per-month features, two graph convolutions
propagate them along the chain, and mean pooling over the window gives the
prediction.

Because the graph is the same fixed chain for every window, the propagation
matrix is constant and the graph convolution reduces to a matmul, so no graph
library is required. This model reads the calendar encodings in addition to the
two satellite predictors.
"""

import torch
import torch.nn as nn

from common import (
    FULL_FEATURES,
    SEQ_LEN,
    TCN,
    Head,
    chain_adjacency,
    run_sequence_models,
)


class GraphConv(nn.Module):
    """One GCN layer on the fixed chain: normalised adjacency, then a linear map."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("adj", chain_adjacency(SEQ_LEN, torch.device("cpu")))

    def forward(self, x):
        return self.linear(self.adj @ x)


class TCNGNN(nn.Module):
    def __init__(self, in_features=len(FULL_FEATURES)):
        super().__init__()
        self.tcn = TCN(in_features, [16, 32])
        self.gnn1 = GraphConv(32, 16)
        self.gnn2 = GraphConv(16, 8)
        self.head = Head(8)

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.relu(self.gnn1(x))
        x = self.gnn2(x)
        return self.head(x.mean(dim=1))


MODELS = {"tcn_gnn": TCNGNN}

if __name__ == "__main__":
    run_sequence_models("TCN-GNN", MODELS, features=FULL_FEATURES)
