# RF / MLP Proxy Experiment

Run:

```powershell
& .\.venv\Scripts\python.exe .\cs-4320-capstone\capstone-experiments\proxy_rf_mlp_experiment\run_rf_mlp_proxy_comparison.py
```

The script compares `Random Forest` and `MLP` on:

- `raw_only`
- `raw_plus_best_proxy`

It saves:

- `comparison_summary.json`
- `comparison_summary.md`
- `comparison_results.csv`
- search-result CSVs for each model / feature-set pair
- confusion-matrix CSVs
- comparison plots
- a PCA plot of the proxy features
