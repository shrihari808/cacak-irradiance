"""Fill the pre-2001 gap in the clearness index and surface albedo.

NASA POWER does not produce ALLSKY_KT or ALLSKY_SRF_ALB before 2001, leaving
204 of the 507 monthly records incomplete. Both columns are reconstructed by
MissForest: initialise at the column means, then repeatedly refit a random
forest for each incomplete column against all the others until the imputed
block stops changing.

The iteration settles into an exact two-cycle rather than a fixed point, so
convergence is tested against the state two steps back as well as one. See
missforest() for why that matters and how the stopping phase is chosen.

Writes the completed table to data/ and the before/after curves to output/.

    python tsmf_imputation.py
    python tsmf_imputation.py --max-iter 100 --tol 1e-6
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

IMPUTE_COLS = ["ALLSKY_KT", "ALLSKY_SRF_ALB"]
FEATURES = [
    "time_idx",
    "month_sin",
    "month_cos",
    "Seconds",
    "ALLSKY_KT",
    "ALLSKY_SRF_ALB",
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
]
SPLIT = "2000-12-31"  # last month before POWER starts producing KT / albedo

# Paths are anchored on this file so the script runs from any directory. The
# completed table is an input to later stages and lands in data/; the figures
# are build products and go to output/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "data", "cacak-monthly.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "cacak-missforest.csv")
DEFAULT_PLOT_DIR = os.path.join(ROOT, "output", "imputation_plots")


def create_df(path):
    read = pd.read_excel if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    df = read(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def seasonal_features(df):
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["time_idx"] = np.arange(len(df))
    df["Seconds"] = df.index.astype("int64") // 1e9
    return df


def plot_series(df, col, title, path):
    """Gap period in orange, satellite-observed period in blue."""
    gap = df.index <= SPLIT
    observed = df.index > SPLIT

    plt.figure(figsize=(14, 6))
    plt.plot(df.index[gap], df[col][gap], color="orange", label="1984–2000")
    plt.plot(df.index[observed], df[col][observed], color="blue", label="2001-2026")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def missforest(df_input, max_iter, tol):
    """Iterative random-forest imputation, initialised at the column means.

    With the forests seeded, a few cells settle into an exact two-cycle instead
    of a fixed point: the change from one iteration to the next stops decaying
    and holds at a constant floor, so a tolerance on that quantity alone can
    never be met. Convergence is therefore measured against the better of the
    last two states, which treats a repeated state as converged whether the
    orbit has period one or two.

    A two-cycle has two phases that differ slightly, so stopping early is only
    equivalent to running the full max_iter if the parity matches. When it does
    not, one further iteration is taken.
    """
    missing = {c: df_input[c].isna() for c in IMPUTE_COLS}
    X = df_input.copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean())

    history = [X[IMPUTE_COLS].to_numpy(copy=True)]
    stopped_at, period = max_iter, None

    for iteration in range(1, max_iter + 1):
        for target_col in IMPUTE_COLS:
            target_idx = X.columns.get_loc(target_col)
            other_indices = [i for i in range(len(X.columns)) if i != target_idx]
            train_rows = ~missing[target_col]
            predict_rows = missing[target_col]

            model = RandomForestRegressor(criterion="friedman_mse", random_state=42, verbose=0)
            model.fit(
                X.iloc[train_rows.values, other_indices],
                X.iloc[train_rows.values, target_idx],
            )
            X.loc[predict_rows, target_col] = model.predict(X.iloc[predict_rows.values, other_indices])

        current = X[IMPUTE_COLS].to_numpy(copy=True)
        d1 = np.abs(current - history[-1]).max()
        d2 = np.abs(current - history[-2]).max() if len(history) >= 2 else np.inf
        history.append(current)

        if iteration <= 5 or iteration % 25 == 0:
            print(f"  iter {iteration:4d}  change vs prev {d1:.3e}   vs two-back {d2:.3e}")

        if tol is not None and min(d1, d2) <= tol:
            period = 1 if d1 <= tol else 2
            # A period-2 orbit has two phases. Stop on the one the iteration cap
            # would have landed on, so the result is identical to running to
            # max_iter; a fixed point has only one phase and can stop at once.
            if period == 2 and iteration % 2 != max_iter % 2:
                continue
            stopped_at = iteration
            print(
                f"  converged at iteration {iteration}: "
                f"{'fixed point' if period == 1 else 'period-2 cycle'} "
                f"(change {min(d1, d2):.3e} <= tol {tol:.1e})"
            )
            break

    return X, stopped_at, period


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="completed monthly table")
    ap.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    ap.add_argument("--max-iter", type=int, default=100, help="iteration cap")
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="converged once the imputed block repeats a state within this tolerance; None disables",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    out = lambda name: os.path.join(args.plot_dir, name)

    df = seasonal_features(create_df(args.input))
    n_missing = {c: int(df[c].isna().sum()) for c in IMPUTE_COLS}
    print(f"loaded {args.input}: {len(df)} months, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}")
    for c in IMPUTE_COLS:
        print(f"  {c:18s} missing {n_missing[c]}")

    for col in IMPUTE_COLS:
        plot_series(df, col, col, out(f"{col}_before.png"))

    print(f"\nMissForest (max_iter={args.max_iter}, tol={args.tol})")
    start = time.time()
    X, stopped_at, period = missforest(df[FEATURES].copy(), args.max_iter, args.tol)
    print(f"done in {time.time() - start:.1f}s after {stopped_at} iterations")
    if period == 2:
        print(f"  result is identical to running the full {args.max_iter} iterations")

    df_imputed = df.copy(deep=True)
    for col in IMPUTE_COLS:
        df_imputed[col] = X[col]

    csv_path = args.output
    df_imputed.to_csv(csv_path, float_format="%.6f")

    for col in IMPUTE_COLS:
        plot_series(df_imputed, col, f"{col} Imputed", out(f"{col}_after.png"))

    print(f"\nwrote {csv_path}")
    print(f"  rows {len(df_imputed)}, columns: {', '.join(df_imputed.columns)}")
    print(f"  remaining NaNs: {int(df_imputed.isna().sum().sum())}")
    for col in IMPUTE_COLS:
        filled = df_imputed.loc[df_imputed.index <= SPLIT, col]
        kept = df_imputed.loc[df_imputed.index > SPLIT, col]
        print(
            f"  {col:18s} imputed {filled.min():.4f}-{filled.max():.4f} (mean {filled.mean():.4f})"
            f"   observed {kept.min():.4f}-{kept.max():.4f} (mean {kept.mean():.4f})"
        )
    print(f"wrote 4 plots to {args.plot_dir}/")


if __name__ == "__main__":
    main()
