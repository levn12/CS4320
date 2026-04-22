# Random Forest vs MLP Proxy Comparison

- Rows: `7861`
- Primary validation metric: `balanced_accuracy`
- Feature sets compared: `raw_only, raw_plus_best_proxy`

## Best Configuration

- Model: `Random Forest`
- Feature set: `raw_plus_best_proxy`
- Test balanced accuracy: `0.9971`
- Test F1: `0.9971`

## Results

```text
        model         feature_set  val_balanced_accuracy  test_balanced_accuracy  test_f1                                                                             best_params
Random Forest raw_plus_best_proxy               0.997797                0.997063 0.997063    {"n_estimators": 500, "max_depth": null, "max_features": 0.5, "min_samples_leaf": 1}
Random Forest            raw_only               0.842637                0.859393 0.858951 {"n_estimators": 500, "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1}
          MLP raw_plus_best_proxy               0.837122                0.843488 0.809322        {"hidden_layer_sizes": [256, 128], "alpha": 0.0001, "learning_rate_init": 0.001}
          MLP            raw_only               0.831359                0.842434 0.811129         {"hidden_layer_sizes": [128, 64], "alpha": 0.0001, "learning_rate_init": 0.001}
```

## Top Random Forest Features

```text
               feature  importance
abs_current_sum_signed    0.245791
           z_proxy_3ph    0.138040
           current_sum    0.092217
       current_abs_max    0.077764
                abs_Ib    0.074271
                abs_Ic    0.068077
                abs_Ia    0.040650
                abs_Va    0.036579
      current_abs_mean    0.031179
    current_vector_mag    0.025140
       abs_current_sum    0.022212
                abs_Vb    0.014289
    voltage_vector_mag    0.012196
             z_proxy_B    0.011694
             z_proxy_A    0.009992
```

## Proxy Features Used

- `abs_Ia`
- `abs_Va`
- `z_proxy_A`
- `abs_Ib`
- `abs_Vb`
- `z_proxy_B`
- `abs_Ic`
- `abs_Vc`
- `z_proxy_C`
- `abs_current_sum`
- `abs_voltage_sum`
- `current_sum`
- `abs_current_sum_signed`
- `current_vector_mag`
- `voltage_vector_mag`
- `z_proxy_3ph`
- `current_abs_mean`
- `current_abs_std`
- `current_abs_max`
- `voltage_abs_mean`
- `voltage_abs_std`
- `current_unbalance`
- `voltage_unbalance`
- `instantaneous_power_proxy`
- `abs_instantaneous_power_proxy`
- `v_i_alignment`
- `z_proxy_AB`
- `z_proxy_BC`
- `z_proxy_CA`