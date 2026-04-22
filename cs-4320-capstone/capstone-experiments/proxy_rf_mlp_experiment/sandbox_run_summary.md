# Sandbox Run Summary

This file records the metrics from a verified sandbox run of
`run_rf_mlp_proxy_comparison.py`.

## Results

```text
        model         feature_set  val_balanced_accuracy  test_balanced_accuracy  test_f1
Random Forest raw_plus_best_proxy               0.997797                0.997063 0.997063
Random Forest            raw_only               0.842637                0.859393 0.858951
          MLP raw_plus_best_proxy               0.837122                0.843488 0.809322
          MLP            raw_only               0.831359                0.842434 0.811129
```

## Notes

- The script itself completed successfully.
- In this sandbox, Python could not write CSV/PNG/JSON outputs into the capstone folder, so the normal saved outputs were not produced here.
- On a normal local run from the workspace `.venv`, the script is configured to save summaries and plots inside the `outputs` folder.
