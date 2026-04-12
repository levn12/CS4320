# Retail Customer Behavior for Unsupervised Learning

## Files

- `retail_customer_behavior_unsupervised.csv` — main dataset

## Intended Use

This is an **unsupervised learning** dataset. It was intentionally designed to support:

- preprocessing with imputation, scaling, and categorical encoding
- PCA variance analysis and 2D visualization
- k-means clustering across multiple values of `k`
- cautious cluster interpretation

## Important Notes

- `customer_id` is an identifier and should be excluded from modeling.

## Column Summary

### Numeric features
- `age`
- `annual_income_k`
- `tenure_months`
- `monthly_orders`
- `avg_basket_usd`
- `discount_share`
- `app_sessions_per_month`
- `website_minutes_per_month`
- `support_tickets_6m`
- `returns_6m`
- `days_since_last_order`
- `delivery_distance_km`
- `satisfaction_score`
- `ad_exposure_score`
- `account_balance_points`

### Categorical features
- `preferred_device`
- `region`
- `membership_tier`
- `primary_category`

