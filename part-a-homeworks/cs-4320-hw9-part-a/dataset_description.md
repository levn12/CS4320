# Synthetic Dataset: Manufacturing Quality Escalation Risk

## Concept and story
This dataset simulates a light-industrial manufacturing setting in which each row represents a production run. The target indicates whether the run ultimately escalated into a quality incident (`target=1`) after inspection and downstream review.

The dataset was designed for an ensemble-learning assignment comparing a single decision tree, a Random Forest, and a boosting model. It intentionally creates conditions where a deep single tree has high variance, Random Forest reduces that variance, and gradient boosting can outperform bagging while also showing overfitting when pushed too hard.

## Feature definitions

| Feature | Type | Description |
|---|---|---|
| risk_signal | numeric | latent process risk indicator |
| process_temp | numeric | centered process temperature deviation |
| pressure_variation | numeric | pressure instability measure |
| vibration_index | numeric | machine vibration intensity |
| operator_experience | numeric | operator experience in years |
| humidity | numeric | shop-floor humidity percentage |
| shift_load | numeric | current workload intensity |
| raw_material_score | numeric | incoming material quality proxy |
| sensor_drift | numeric | sensor drift measure partly correlated with risk |
| maintenance_gap_days | numeric | days since last meaningful maintenance |
| ambient_temp | numeric | ambient air temperature |
| weak_proxy | numeric | weak noisy summary feature |
| random_id_hash | numeric | essentially spurious numeric field |
| machine_type | categorical | machine family A-D |
| supplier_tier | categorical | supplier quality tier |
| shift | categorical | day / swing / night shift |
| region | categorical | plant region |
| target | binary | 1 = quality incident escalation |


## Recommended split
Use a stratified **70 / 15 / 15** train / validation / test split.

For `n=3200`, that gives approximately:
- Train: 2240
- Validation: 480
- Test: 480

