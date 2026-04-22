# Capstone final run summary

## Dataset
- Source CSV: `C:\Users\levid\School_Programming\CS4320\cs-4320-capstone\electrical_fault_data.csv`
- Rows: `7861`
- Classes: `6`
- Fault pattern counts: `0000`=2365, `0110`=1004, `0111`=1096, `1001`=1129, `1011`=1134, `1111`=1133

## Feature-set benchmark
- `raw_plus_proxy`: validation balanced accuracy `0.9971`
- `proxy_only`: validation balanced accuracy `0.9971`
- `raw_only`: validation balanced accuracy `0.8449`

## Final model
- Model family: `RandomForestClassifier`
- Final feature representation: `raw_plus_proxy`
- Best hyperparameters: `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 300}`
- Out-of-bag score on the training split: `0.9981`
- Actual fitted tree depths: min `10`, median `15.0`, mean `15.48`, max `25`
- Actual fitted leaf counts: min `63`, median `108.0`, mean `111.86`, max `183`

## Metrics
- Train metrics: accuracy `1.0000`, balanced accuracy `1.0000`, macro F1 `1.0000`
- Test metrics: accuracy `0.9968`, balanced accuracy `0.9963`, macro precision `0.9963`, macro recall `0.9963`, macro F1 `0.9963`

## Most important features
- `abs_current_sum_signed` (proxy): `0.3199`
- `z_proxy_3ph` (proxy): `0.0974`
- `current_vector_mag` (proxy): `0.0907`
- `z_proxy_A` (proxy): `0.0822`
- `voltage_vector_mag` (proxy): `0.0786`
- `current_unbalance` (proxy): `0.0637`
- `z_proxy_B` (proxy): `0.0584`
- `z_proxy_C` (proxy): `0.0389`
- `abs_instantaneous_power_proxy` (proxy): `0.0360`
- `Ib` (raw): `0.0285`
