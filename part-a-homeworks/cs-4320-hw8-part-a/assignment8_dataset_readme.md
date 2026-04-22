# CS 4320 --- Assignment 8 (Part A)

## Dataset: Subscription Churn Prediction

**File:** `cs4320_assignment8_svm_dataset.csv`\
**Target column:** `churned_next_month` (binary: 0 or 1)

------------------------------------------------------------------------

## Scenario

You are working with a subscription-based software company.

Each row in this dataset represents a customer snapshot at the end of a
given month. Your task is to build a model that predicts whether the
customer will **churn (cancel their subscription) next month**.

This dataset contains a mix of:

-   Numeric features\
-   Categorical features\
-   Features on different scales\
-   Some missing values

------------------------------------------------------------------------

## Columns

### Identifier

-   **`customer_id`**\
    Unique identifier for each customer.\
    This column should **not** be used as a predictive feature.

------------------------------------------------------------------------

### Numeric Features

-   **`engagement_score`** (float)\
    Composite engagement metric based on user activity.

-   **`billing_stability`** (float)\
    Proxy metric for billing reliability.

-   **`tenure_months`** (int)\
    Number of months the customer has been subscribed.

-   **`avg_session_seconds`** (int)\
    Average session duration in seconds.

-   **`monthly_spend_usd`** (float)\
    Monthly spending in USD.

-   **`support_tickets_90d`** (int)\
    Number of support tickets submitted in the past 90 days.

-   **`late_payments_12m`** (int)\
    Number of late payments in the past 12 months.

-   **`marketing_emails_opened_30d`** (int)\
    Number of marketing emails opened in the past 30 days.

------------------------------------------------------------------------

### Categorical Features

-   **`region`**\
    One of: `West`, `Mountain`, `Midwest`, `South`, `Northeast`

-   **`plan_type`**\
    One of: `Basic`, `Standard`, `Pro`, `Enterprise`

-   **`primary_device`**\
    One of: `Mobile`, `Desktop`, `Tablet`

------------------------------------------------------------------------

### Target

-   **`churned_next_month`**
    -   `1` → customer churned next month\
    -   `0` → customer retained next month

------------------------------------------------------------------------

## Important Notes

-   Do not assume all features are on the same scale.
-   Do not perform preprocessing outside your Pipeline.
-   Do not use `customer_id` as a feature.
-   Your final model selection must be based on validation performance.
-   The test set should be used **once** for final evaluation.

------------------------------------------------------------------------

## Academic Integrity Reminder

You may discuss high-level modeling strategy with classmates, but:

-   Your code must be your own.
-   Your analysis and written interpretation must be your own.
-   Do not share trained models or prediction files.
