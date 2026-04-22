# Selected Proxy RF vs MLP Comparison

- Rows: `7861`
- Primary validation metric: `balanced_accuracy`

## Proxy Features Used

- `z_proxy_A`
- `z_proxy_B`
- `z_proxy_C`
- `z_proxy_AB`
- `z_proxy_BC`
- `z_proxy_CA`
- `z_proxy_3ph`
- `current_vector_mag`
- `voltage_vector_mag`
- `abs_current_sum_signed`
- `current_unbalance`
- `voltage_unbalance`
- `v_i_alignment`
- `abs_instantaneous_power_proxy`

## Best Configuration

- Model: `Random Forest`
- Test balanced accuracy: `0.9941`
- Test F1: `0.9941`

## Results

```text
        model  val_balanced_accuracy  test_balanced_accuracy  test_f1                                                                             best_params
Random Forest               0.995561                0.994126 0.994083 {"n_estimators": 300, "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1}
          MLP               0.864069                0.849959 0.829530         {"hidden_layer_sizes": [128, 64], "alpha": 0.0001, "learning_rate_init": 0.001}
```

## Random Forest Importances

```text
                      feature  importance
       abs_current_sum_signed    0.235225
           current_vector_mag    0.122632
                  z_proxy_3ph    0.115037
           voltage_vector_mag    0.097997
                    z_proxy_A    0.077054
            current_unbalance    0.060403
                    z_proxy_B    0.059209
                   z_proxy_CA    0.051909
                    z_proxy_C    0.044145
abs_instantaneous_power_proxy    0.042572
                v_i_alignment    0.028787
                   z_proxy_BC    0.027791
                   z_proxy_AB    0.019277
            voltage_unbalance    0.017962
```