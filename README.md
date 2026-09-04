# Solar irradiance modelling for the Čačak grid cell

Code accompanying the manuscript on low-cost weather-station irradiance
measurement and satellite-series regression for Čačak, Serbia.

This repository covers the machine learning half of the study: building the
monthly NASA POWER series for the study grid cell, reconstructing the two
predictor columns that the satellite product does not provide before 2001, and
training and evaluating 27 regression models on the completed series.

## Data

All values come from the NASA POWER daily point API for 43.89° N, 20.35° E, the
0.5° grid cell containing all three measurement sites. Daily values are averaged
to calendar months, giving 507 months from January 1984 to March 2026.

| column | role | availability |
| --- | --- | --- |
| `ALLSKY_KT` | predictor, all-sky insolation clearness index | 2001 onward |
| `ALLSKY_SRF_ALB` | predictor, all-sky surface albedo | 2001 onward |
| `ALLSKY_SFC_SW_DWN` | target, all-sky shortwave irradiance | 1984 onward |
| `CLRSKY_SFC_SW_DWN` | target, clear-sky shortwave irradiance | 1984 onward |

POWER does not produce the clearness index or the surface albedo before 2001, so
303 of the 507 months are complete and 204 are missing both predictors. Those
are filled by MissForest before modelling.

The daily API is the source used throughout. POWER also publishes a wide
"Monthly and Annual" export which can be reshaped offline, but that product
rounds the clearness index and albedo to two decimals, collapsing roughly 87% of
their distinct values, and it lags the daily archive. It is supported only as a
fallback.

## Layout

```
data/
  cacak-unprocessed.csv       raw POWER export, if working offline
  cacak-monthly.csv           monthly series, 204 months with gaps
  cacak-missforest.csv        completed series, model input
output/
  checkpoints/                best-validation weights per model
  imputation_plots/           predictor curves before and after imputation
  results/
    results.csv               all 27 models
    results_consolidated.csv  best model per family
src/
  build_cacak_monthly.py      POWER retrieval and monthly aggregation
  tsmf_imputation.py          MissForest reconstruction of the pre-2001 gap
  common.py                   shared configuration, data pipeline, metrics
  <family>.py                 one script per architecture family
  consolidate_results.py      best model per family
```

## Requirements

Python 3.10 or newer.

```
pip install numpy pandas scikit-learn matplotlib requests torch xgboost
```

A GPU is not needed. The full set of models trains in a few minutes on CPU.

## Reproducing the results

Run in order from the repository root. Every script resolves its own paths, so
the working directory does not matter.

```
python src/build_cacak_monthly.py --source daily
python src/tsmf_imputation.py
```

Then any subset of the model scripts. Each is independent and appends to the
same results file, so they can be run in any order or one at a time.

```
python src/xgboost_model.py   python src/cnn_lstm.py      python src/lstm_gru.py
python src/ann.py             python src/cnn_gru.py       python src/tcn_gnn.py
python src/cnn.py             python src/tcn_lstm.py
python src/tcn.py             python src/tcn_gru.py
python src/lstm.py
python src/gru.py
```

Finally:

```
python src/consolidate_results.py
```

Re-running a model script replaces that model's row rather than appending a
duplicate.

## Models

27 models in 12 families. Depth variants are numbered by suffix.

| family | script | variants | architecture |
| --- | --- | --- | --- |
| XGBoost | `xgboost_model.py` | 1 | gradient-boosted trees, multi-output |
| ANN | `ann.py` | 3 | flattened window, 64, 64-32 or 64-32-16 dense ReLU |
| CNN | `cnn.py` | 3 | one to three 1D convolutions, global average pooling |
| TCN | `tcn.py` | 3 | one to three dilated causal blocks |
| LSTM | `lstm.py` | 3 | 64, 64-32, 64-32-16 |
| GRU | `gru.py` | 3 | 64, 64-32, 64-32-16 |
| CNN-LSTM | `cnn_lstm.py` | 2 | convolutional front end, 32 or 32-16 recurrent |
| CNN-GRU | `cnn_gru.py` | 2 | as above with GRU cells |
| TCN-LSTM | `tcn_lstm.py` | 2 | dilated causal front end, 32 or 32-16 recurrent |
| TCN-GRU | `tcn_gru.py` | 2 | as above with GRU cells |
| LSTM-GRU | `lstm_gru.py` | 2 | 64 LSTM into 32 or 32-16 GRU |
| TCN-GNN | `tcn_gnn.py` | 1 | temporal convolutions, then graph convolutions over the window |

Every neural model ends in the same head: dropout, a 16-unit dense layer with
leaky ReLU, and a linear layer producing both targets.

## Training protocol

Identical for every model, so that only the architecture varies.

| | |
| --- | --- |
| input window | 12 months |
| split | chronological 80/20, no shuffling |
| validation | further 80/20 split of the training portion |
| scaling | min-max on features and targets |
| optimiser | Adam, learning rate 1e-3, no weight decay |
| loss | mean squared error |
| epochs | 100, weights restored from the best validation epoch |
| batch size | 32 |
| seed | 42, reset before each model |

Metrics are computed on inverse-scaled values. MAE, MAPE, RMSE and R² are the
headline figures; MSE, MSLE, RMSLE and adjusted R² are recorded alongside them.

## Implementation notes

**Feature sets differ by model.** The sequence models are trained on the two
satellite predictors alone. The graph and gradient-boosting models additionally
receive the calendar encodings (`month_sin`, `month_cos`, `time_idx`,
`Seconds`). This asymmetry is inherited from the original experiment and is kept
so that results remain comparable with the published ones. The `n_features`
column in `results.csv` records which set each model used, and it is also the
`j` term in the adjusted R² denominator.

**XGBoost sees a different test set.** It is fitted on the flat monthly table
rather than on sliding windows, so its test partition is the last fifth of
months rather than the last fifth of windows. Its `n_test` is correspondingly
larger.

**The imputation does not reach a fixed point.** A few cells settle into an
exact two-cycle, so the change between consecutive iterations stops decaying and
holds at a constant floor. Convergence is therefore tested against the better of
the last two states. Because the two phases of the cycle differ slightly, the
stopping iteration is chosen to match the parity of `--max-iter`, which makes an
early stop numerically identical to running the full iteration count.

**No graph library is required.** Each 12-month window is the same fixed path
graph, so the GCN propagation matrix is constant and the graph convolution
reduces to a matrix multiplication.

**The temporal blocks are not keras-tcn.** They are plain dilated causal
convolutions without residual connections or weight normalisation. TCN results
are internally consistent but should not be compared directly against
keras-tcn numbers.

**Edge footprint.** `results.csv` records parameter counts and model sizes for
every model. TCN-GNN reaches R² = 0.9586 with 2,106 parameters, roughly 2.1 KB
of INT8 weights, and its graph step reduces to a constant 12x12 matmul, so it
needs no graph library at inference. That makes the second-best model in the
comparison a viable microcontroller candidate.

## Results

Best model per family on the held-out final fifth of the record.

| family | model | MAE | MAPE (%) | RMSE | R² |
| --- | --- | --- | --- | --- | --- |
| XGBoost | xgboost | 0.0850 | 2.12 | 0.1297 | 0.9954 |
| TCN-GNN | tcn_gnn | 0.2730 | 7.11 | 0.3833 | 0.9586 |
| CNN-GRU | cnn_gru_2 | 0.4509 | 12.36 | 0.5773 | 0.9129 |
| TCN-LSTM | tcn_lstm_1 | 0.4816 | 12.98 | 0.6126 | 0.9017 |
| CNN-LSTM | cnn_lstm_2 | 0.4767 | 12.54 | 0.6186 | 0.9011 |
| TCN-GRU | tcn_gru_2 | 0.4907 | 13.75 | 0.6211 | 0.9005 |
| CNN | cnn_3 | 0.5261 | 15.41 | 0.6705 | 0.8843 |
| ANN | ann_3 | 0.5320 | 13.96 | 0.6702 | 0.8839 |
| GRU | gru_2 | 0.5144 | 14.12 | 0.6716 | 0.8836 |
| TCN | tcn_2 | 0.4952 | 13.03 | 0.6736 | 0.8819 |
| LSTM-GRU | lstm_gru_1 | 0.5330 | 16.07 | 0.6774 | 0.8811 |
| LSTM | lstm_3 | 0.5425 | 14.72 | 0.7145 | 0.8693 |

MAE and RMSE are in kWh/m²/day; MAPE is scale-free. Ranking is by R².
MSE, MSLE, RMSLE and adjusted R² are also recorded for every model in
`output/results/results.csv`, along with parameter counts and timings.

Gradient boosting leads by a wide margin, which is the expected outcome for a
moderate-size tabular problem with strong seasonal structure. Among the neural
models the graph architecture is clearly ahead.

The ANN baseline is the result worth dwelling on. `ann_3` reaches R² = 0.8839
with 4,514 parameters and no architectural representation of temporal order,
which matches the best CNN and beats the best GRU, TCN, LSTM-GRU and LSTM. Only
XGBoost, TCN-GNN and the four convolutional-recurrent hybrids separate
themselves from a plain feed-forward network by a clear margin. At monthly
resolution the seasonal cycle is evidently recoverable from a flattened window,
so architectural complexity is not by itself rewarded here.

## Limitations

The reconstructed 1984-2000 predictor values have visibly narrower spread than
the observed period, since averaging over forest trees pulls estimates toward
the mean. Surface albedo reaches 0.41 in the observed record but only 0.32 in
the reconstructed block. Results that depend on variance or on extremes in the
early period should be read with that in mind.
