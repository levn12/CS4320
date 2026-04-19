# Electrical Fault Data Inspection

## Dataset Summary

- Source file: `electrical_fault_data.csv`
- Rows: `7861`
- Measurement features: `Ia, Ib, Ic, Va, Vb, Vc`
- Fault indicator columns: `G, C, B, A`
- Unique fault patterns: `0000, 0110, 0111, 1001, 1011, 1111`
- Rows with any fault: `5496`
- Rows with no fault: `2365`

## Fault Pattern Counts

- `0000`: `2365`
- `0110`: `1004`
- `0111`: `1096`
- `1001`: `1129`
- `1011`: `1134`
- `1111`: `1133`

## Feature Summary

```text
            Ia        Ib        Ic      Va      Vb      Vc
mean   13.7212  -44.8453   34.3924 -0.0077  0.0012  0.0065
std   464.7417  439.2692  371.1074  0.2892  0.3134  0.3079
min  -883.5423 -900.5270 -883.3578 -0.6207 -0.6080 -0.6127
max   885.7386  889.8689  901.2743  0.5953  0.6279  0.6002
```

## PCA Notes

- PC1 explained variance: `30.74%`
- PC2 explained variance: `25.70%`
- First two PCs combined: `56.44%`

## Strong Feature Correlations

- `Vb` vs `Vc`: `-0.567`
- `Ib` vs `Ic`: `-0.528`
- `Va` vs `Vb`: `-0.480`
- `Va` vs `Vc`: `-0.450`
- `Ia` vs `Ib`: `-0.374`
- `Ia` vs `Ic`: `-0.276`

## Quick Interpretation

- The current channels and voltage channels both show strong structure rather than random scatter.
- The PCA projection is useful for seeing that the fault patterns occupy related but only partially separated regions.
- The per-phase voltage-vs-current plots help show that the data behaves more like structured electrical states than isolated independent measurements.
- The standardized mean-profile plot helps show how each fault pattern shifts the six measurements in a consistent way.

## Generated Files

- `class_balance.png`
- `pca_projection_2d.png`
- `pca_explained_variance.png`
- `currents_3d.png`
- `voltages_3d.png`
- `phase_a_voltage_vs_current.png`
- `phase_voltage_current_grid.png`
- `feature_correlation_heatmap.png`
- `feature_distributions.png`
- `fault_pattern_mean_profile.png`
- `data_readme.md`