# Capstone Final Model

This folder contains a polished end-to-end capstone deliverable for multiclass electrical fault classification with a tuned Random Forest. The final model uses both the six raw sensor readings (`Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`) and a set of impedance- and phase-related proxy features engineered from those readings.

## What the script does

The main script is `run_capstone_final_model.py`.

Its workflow is:

1. Load `electrical_fault_data.csv` and build the multiclass label from the four fault bits `G`, `C`, `B`, and `A`.
2. Build raw, proxy-only, and raw-plus-proxy feature representations.
3. Generate descriptive plots for the raw features and the proxy features, including a PCA plot for each representation.
4. Benchmark the three feature sets with a shared validation split and Random Forest models.
5. Tune a final Random Forest on the combined raw-plus-proxy feature set with the same validation split.
6. Fit the chosen model on the training split and evaluate once on a held-out test split.
7. Save metrics, confusion matrices, feature importances, predictions, a serialized model artifact, and a run summary.

## How to run it

From the repository root:

```powershell
python cs-4320-capstone/capstone-final/run_capstone_final_model.py
```

Optional arguments:

```powershell
python cs-4320-capstone/capstone-final/run_capstone_final_model.py --data-path path\to\electrical_fault_data.csv --output-dir path\to\outputs
```

If `--data-path` is omitted, the script uses the CSV one folder up from this script.

- `cs-4320-capstone/electrical_fault_data.csv`

## Output structure

Running the script creates an `outputs` folder with three subfolders:

- `plots/`: data visualizations, PCA plots, confusion matrices, feature importance charts, and model-performance plots
- `tables/`: metrics, CV search results, class distributions, predictions, feature summaries, and importance tables
- `model/`: the serialized trained pipeline and metadata JSON

It also writes `run_summary.md`, which is the quickest place to read the main results from the latest run.

## Important modeling choices

- The target is the multiclass fault pattern, not the easier binary `any fault` target.
- The fault-indicator columns are used only to create the label. They are never included as model inputs, which prevents label leakage.
- The final model is a Random Forest because it fits this problem well: the feature space is numeric, moderately low-dimensional, and likely contains nonlinear interactions.
- Balanced accuracy is the primary tuning metric so that performance is not dominated by the largest class.
- The workflow uses a leakage-safe train/validation/test split. Hyperparameter tuning is done only on the training split and checked on validation, and the test split stays untouched until the end.

## Proxy-feature considerations

The proxy features are not true impedance or phasor calculations from a full power-systems model. They are practical approximations built from the available snapshot data. That means they should be interpreted as physically motivated diagnostics rather than exact electrical quantities.

Examples include:

- per-phase voltage-to-current magnitude ratios
- three-phase magnitude ratios
- phase-to-phase current and voltage difference ratios
- current and voltage unbalance measures
- vector magnitude, alignment, and power-style proxies

These features are useful because they expose relationships between the raw readings that a tree model can exploit more directly.

## Files worth checking first

- `run_capstone_final_model.py`
- `outputs/run_summary.md`
- `outputs/tables/final_metrics.csv`
- `outputs/plots/top_feature_importance.png`
- `outputs/plots/test_confusion_matrix_normalized.png`
- `outputs/plots/raw_pca_scatter.png`
- `outputs/plots/proxy_pca_scatter.png`

## Notes

- The script is meant to be rerunnable and self-contained.
- If you rerun it, the `outputs` folder will be refreshed with a new set of artifacts from that run.
- The serialized model is a full sklearn pipeline, so loading it later will include preprocessing and the trained Random Forest together.
