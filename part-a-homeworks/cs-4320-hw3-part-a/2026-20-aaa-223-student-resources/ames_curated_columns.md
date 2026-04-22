# Ames Housing (Curated) — Merged Column Guide

This document merges the assignment data dictionary and student column guide for
`ames_curated.csv`.

Dataset shape: 2,930 rows × 15 columns.

Notes:
- Some columns contain missing values (commonly in garage/basement fields and a
  small number of numeric fields). You will handle these as part of Assignment 3.
- `saleprice` is the prediction target.
- You will decide which remaining columns are appropriate model inputs.

## Columns

| Column        | Type        | Description                                                   |
|---------------|-------------|---------------------------------------------------------------|
| pid           | integer     | Row identifier for the property record.                       |
| neighborhood  | categorical | Neighborhood name (broad location within Ames).               |
| overall_qual  | integer     | Overall material/finish quality rating (higher is better).    |
| year_built    | integer     | Year the home was originally constructed.                     |
| lot_area      | numeric     | Lot size in square feet.                                      |
| gr_liv_area   | numeric     | Above-ground living area in square feet.                      |
| total_bsmt_sf | numeric     | Total basement area in square feet.                           |
| full_bath     | integer     | Number of full bathrooms.                                     |
| bedroom_abvgr | integer     | Bedrooms above ground.                                        |
| garage_cars   | integer     | Garage capacity (number of cars).                             |
| garage_type   | categorical | Type of garage (e.g., attached/detached/none).                |
| kitchen_qual  | categorical | Kitchen quality rating (ordinal categories).                  |
| bsmt_qual     | categorical | Basement quality rating (ordinal categories).                 |
| saleprice     | numeric     | Target: sale price of the home (USD).                         |
| pricepersf    | numeric     | Precomputed ratio feature (USD per square foot).              |
