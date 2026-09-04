"""Reduce the full results table to the best model per architecture family.

results.csv holds every variant that was trained: three LSTM depths, three CNN
depths, two of each hybrid and so on. This keeps only the strongest variant of
each family, which is the form the manuscript reports.

Ranking is by RANK_METRIC. Set RANK_ASCENDING to True when switching to an
error metric such as RMSE, where smaller is better.
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "output", "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "results.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "results_consolidated.csv")

RANK_METRIC = "R2"
RANK_ASCENDING = False  # False: larger is better (R2). True: smaller (RMSE, MAE).

COLUMNS = [
    "family", "model", "MAE", "MAPE(%)", "MSE", "RMSE", "MSLE", "RMSLE",
    "R2", "Adj_R2", "params", "train_time_s", "inf_time_s", "n_test", "n_features",
]


def main():
    if not os.path.exists(INPUT_CSV):
        sys.exit(f"No results at {INPUT_CSV}. Run the model scripts first.")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        sys.exit(f"{INPUT_CSV} has no rows.")

    missing = [c for c in ("family", "model", RANK_METRIC) if c not in df.columns]
    if missing:
        sys.exit(f"{INPUT_CSV} is missing required columns: {', '.join(missing)}")

    # idxmin/idxmax return a single row per family even when variants tie.
    pick = df.groupby("family")[RANK_METRIC]
    best_idx = pick.idxmin() if RANK_ASCENDING else pick.idxmax()
    best = df.loc[best_idx].sort_values(RANK_METRIC, ascending=RANK_ASCENDING)

    best = best[[c for c in COLUMNS if c in best.columns]]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best.to_csv(OUTPUT_CSV, index=False)

    print(f"read   {INPUT_CSV}: {len(df)} models across {df.family.nunique()} families")
    print(f"wrote  {OUTPUT_CSV}: {len(best)} rows, best per family by {RANK_METRIC}\n")

    width = max(len(f) for f in best.family)
    for row in best.itertuples():
        dropped = df[(df.family == row.family) & (df.model != row.model)].model.tolist()
        beat = f"  (over {', '.join(dropped)})" if dropped else ""
        print(f"  {row.family:<{width}}  {row.model:<12} {RANK_METRIC} {getattr(row, RANK_METRIC):.4f}{beat}")


if __name__ == "__main__":
    main()
