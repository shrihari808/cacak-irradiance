"""Shared configuration, data pipeline and result logging.

Every model script imports from this module, so the training protocol is
identical across all 24 models and only the architecture differs. Each script
runs standalone and writes its rows to RESULTS_CSV.

Protocol: chronological 80/20 train/test split with no shuffling, a further
80/20 split of the training portion for validation, min-max scaling fitted on
the full series, 100 epochs of Adam with MSE loss, and the weights from the
best validation epoch restored before the test set is touched. Metrics are
computed on inverse-scaled values.

Two conventions are worth stating explicitly. The graph model needs no
torch_geometric: on a fixed 12-node chain, GCN propagation is a constant
normalised adjacency matmul (chain_adjacency). And the temporal blocks here are
plain dilated causal convolutions without residual connections or weight
normalisation, so they are internally consistent but not interchangeable with
keras-tcn.
"""

import csv
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Anchored on this file rather than the working directory, so the model scripts
# run correctly from src/, from the project root, or from anywhere else.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")

DATA_PATH = os.path.join(DATA_DIR, "cacak-missforest.csv")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

# --------------------------------------------------------------------------
# Hyperparameters, uniform across every model
# --------------------------------------------------------------------------
SEED = 42
SEQ_LEN = 12
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # carved from the training portion, chronological
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.2
DENSE_UNITS = 16
N_OUTPUTS = 2
LEAKY_SLOPE = 0.2  # torch defaults to 0.01; 0.2 matches the reference implementation
CONV_KERNEL = 3
TCN_KERNEL = 2
TCN_LEVELS = 3  # dilations 1,2,4 per TCN layer -> receptive field 8 of 12 months

TARGETS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"]

# The sequence models are trained on the two satellite predictors alone. The
# graph and gradient-boosting models additionally receive the calendar
# encodings. This asymmetry is inherited from the original experiment and is
# kept so the two sets of results stay comparable with the published ones.
SEQ_FEATURES = ["ALLSKY_KT", "ALLSKY_SRF_ALB"]
FULL_FEATURES = ["time_idx", "month_sin", "month_cos", "Seconds", "ALLSKY_KT", "ALLSKY_SRF_ALB"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_frame(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").dropna()


def make_sequences(X, y, seq_len=SEQ_LEN):
    """Sliding windows of seq_len months predicting the following month."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        window = X[i : i + seq_len]
        if np.isnan(window).any():
            continue
        Xs.append(window)
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def sequence_data(features=SEQ_FEATURES):
    """Chronological train/val/test tensors plus the target scaler."""
    df = load_frame()
    X = df[features].to_numpy()
    y = df[TARGETS].to_numpy()

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_s = scaler_x.fit_transform(X)
    y_s = scaler_y.fit_transform(y)

    X_seq, y_seq = make_sequences(X_s, y_s)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, shuffle=False
    )
    cut = int((1 - VAL_SIZE) * len(X_train))
    return {
        "X_tr": X_train[:cut], "y_tr": y_train[:cut],
        "X_val": X_train[cut:], "y_val": y_train[cut:],
        "X_test": X_test, "y_test": y_test,
        "scaler_y": scaler_y,
        "n_features": len(features),
    }


def loaders(data):
    def make(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        make(data["X_tr"], data["y_tr"], False),
        make(data["X_val"], data["y_val"], False),
        make(data["X_test"], data["y_test"], False),
    )


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class Chomp1d(nn.Module):
    """Trim the right padding so the convolution stays causal."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size] if self.chomp_size else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TCN(nn.Module):
    """Stack of dilated causal convolutions; dilation doubles per level."""

    def __init__(self, num_inputs, num_channels, kernel_size=TCN_KERNEL):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 2 ** i))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # (batch, channels, time)
        return self.network(x)


class ConvFront(nn.Module):
    """Conv1d(64, same) -> leaky-ReLU -> MaxPool(2), the hybrid front end."""

    def __init__(self, in_features, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, out_channels, CONV_KERNEL, padding=CONV_KERNEL // 2),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.MaxPool1d(2),
        )
        self.out_features = out_channels

    def forward(self, x):  # (batch, time, features) -> (batch, time/2, channels)
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class TCNFront(nn.Module):
    """Dilated causal stack returning the full sequence, for TCN-RNN hybrids."""

    def __init__(self, in_features, out_channels=64):
        super().__init__()
        self.tcn = TCN(in_features, [out_channels] * TCN_LEVELS)
        self.out_features = out_channels

    def forward(self, x):  # (batch, time, features) -> (batch, time, channels)
        return self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)


class Head(nn.Module):
    """Dropout -> Dense(16, leaky-ReLU) -> Linear(2), shared by every model."""

    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, DENSE_UNITS),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Linear(DENSE_UNITS, N_OUTPUTS),
        )

    def forward(self, x):
        return self.net(x)


class Recurrent(nn.Module):
    """LSTM/GRU stack returning the last hidden state. cells: 'lstm' or 'gru'."""

    def __init__(self, input_size, hidden_sizes, cells):
        super().__init__()
        if isinstance(cells, str):
            cells = [cells] * len(hidden_sizes)
        layers = []
        for size, cell in zip(hidden_sizes, cells):
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            layers.append(rnn(input_size, size, batch_first=True))
            input_size = size
        self.layers = nn.ModuleList(layers)
        self.out_features = hidden_sizes[-1]

    def forward(self, x):  # (batch, time, features)
        for layer in self.layers:
            x, _ = layer(x)
        return x[:, -1, :]


def chain_adjacency(n_nodes, device):
    """Symmetric normalised adjacency of a bidirectional chain with self-loops.

    D^-1/2 (A + I) D^-1/2 for a path graph. Propagating with this matrix is
    equivalent to a GCN layer over the window, without the graph library.
    """
    A = torch.eye(n_nodes, device=device)
    idx = torch.arange(n_nodes - 1, device=device)
    A[idx, idx + 1] = 1.0
    A[idx + 1, idx] = 1.0
    d_inv_sqrt = A.sum(1).pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, name):
    """Train for EPOCHS, restoring the weights from the best validation epoch."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINT_DIR, f"{name}.pt")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best = float("inf")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += criterion(model(xb), yb).item() * len(xb)
        val_loss = total / len(val_loader.dataset)

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), ckpt)
        if epoch % 25 == 0:
            print(f"    epoch {epoch:3d}  val_loss {val_loss:.6f}  (best {best:.6f})")

    train_time = time.time() - start
    model.load_state_dict(torch.load(ckpt))
    return model, train_time, os.path.getsize(ckpt) / 1024


def predict(model, test_loader):
    model.eval()
    preds = []
    start = time.time()
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds), time.time() - start


def metrics(y_true, y_pred, n_features):
    """The eight reported metrics, computed on inverse-scaled values.

    MAPE guards against division by zero and the logarithmic metrics clip
    negative predictions, both of which occur for the weaker models.
    """
    eps = 1e-7
    safe = np.where(y_true == 0, eps, y_true)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(np.clip(y_true, 0, None), np.clip(y_pred, 0, None))
    r2 = r2_score(y_true, y_pred)
    n = y_true.shape[0]
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE(%)": float(np.mean(np.abs((y_true - y_pred) / safe)) * 100),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MSLE": msle,
        "RMSLE": float(np.sqrt(msle)),
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - n_features - 1),
        "n_test": n,
        "n_features": n_features,
    }


# --------------------------------------------------------------------------
# Shared results file
# --------------------------------------------------------------------------
RESULT_FIELDS = [
    "model", "family", "params", "train_time_s", "inf_time_s", "model_size_kb",
    "MAE", "MAPE(%)", "MSE", "RMSE", "MSLE", "RMSLE", "R2", "Adj_R2",
    "n_test", "n_features", "timestamp",
]


def log_result(row):
    """Write one model's row into the shared results file.

    The file accumulates all 24 models whatever order the scripts are run in.
    Re-running a script replaces that model's row instead of duplicating it.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    row = {**row, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("model") != row["model"]]
    rows.append({k: row.get(k, "") for k in RESULT_FIELDS})

    tmp = RESULTS_CSV + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, RESULTS_CSV)  # atomic, so parallel scripts cannot truncate


def run_sequence_models(family, builders, features=SEQ_FEATURES):
    """Train, evaluate and log every variant in one architecture family.

    builders maps each model name to a zero-argument constructor. The seed is
    reset before each variant so results do not depend on execution order.
    """
    set_seed()
    data = sequence_data(features)
    train_loader, val_loader, test_loader = loaders(data)
    y_true = data["scaler_y"].inverse_transform(data["y_test"])

    print(f"{family}: {len(data['X_tr'])} train / {len(data['X_val'])} val / "
          f"{len(data['X_test'])} test windows, {data['n_features']} features, device {DEVICE}")

    for name, build in builders.items():
        print(f"\n  {name}")
        set_seed()
        model = build()
        n_params = sum(p.numel() for p in model.parameters())
        model, train_time, size_kb = train_model(model, train_loader, val_loader, name)
        y_pred_s, inf_time = predict(model, test_loader)
        y_pred = data["scaler_y"].inverse_transform(y_pred_s)

        result = metrics(y_true, y_pred, data["n_features"])
        log_result({
            "model": name, "family": family, "params": n_params,
            "train_time_s": round(train_time, 3), "inf_time_s": round(inf_time, 4),
            "model_size_kb": round(size_kb, 1), **result,
        })
        print(f"    params {n_params:,}  R2 {result['R2']:.4f}  RMSE {result['RMSE']:.4f}  "
              f"MAE {result['MAE']:.4f}  ({train_time:.1f}s)")

    print(f"\n{family} done -> {RESULTS_CSV}")
