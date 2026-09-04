"""Gradient-boosted trees, wrapped for multi-output regression.

The only non-neural model in the comparison. It sees the flat monthly table
rather than 12-month windows, so its test set is the last fifth of months
instead of the last fifth of windows, and its row in results.csv therefore
carries a slightly larger n_test than the sequence models.
"""

import pickle
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from common import (
    RESULTS_CSV,
    SEED,
    TARGETS,
    TEST_SIZE,
    load_frame,
    log_result,
    metrics,
    set_seed,
)


def main():
    set_seed()
    df = load_frame()
    X = df.drop(columns=TARGETS)
    y = df[TARGETS].to_numpy()

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_s = scaler_x.fit_transform(X.to_numpy())
    y_s = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=TEST_SIZE, shuffle=False
    )
    print(f"XGBoost: {len(X_train)} train / {len(X_test)} test months, "
          f"{X.shape[1]} features ({', '.join(X.columns)})")

    model = MultiOutputRegressor(
        XGBRegressor(eval_metric=["mae", "rmse"], random_state=SEED, verbosity=0)
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred_s = model.predict(X_test)
    inf_time = time.time() - start

    y_pred = scaler_y.inverse_transform(y_pred_s)
    y_true = scaler_y.inverse_transform(y_test)

    result = metrics(y_true, y_pred, X.shape[1])
    log_result({
        "model": "xgboost",
        "family": "XGBoost",
        # Trees rather than weights, so that the column stays meaningful here.
        "params": sum(len(e.get_booster().get_dump()) for e in model.estimators_),
        "train_time_s": round(train_time, 3),
        "inf_time_s": round(inf_time, 4),
        "model_size_kb": round(sys.getsizeof(pickle.dumps(model)) / 1024, 1),
        **result,
    })
    print(f"  R2 {result['R2']:.4f}  RMSE {result['RMSE']:.4f}  MAE {result['MAE']:.4f}  "
          f"({train_time:.1f}s)")
    print(f"\nXGBoost done -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
