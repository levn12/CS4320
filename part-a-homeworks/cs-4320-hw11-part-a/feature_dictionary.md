# Feature Dictionary

| Feature | Type | Description | Likely role |
|---|---|---|---|
| tenure_months | numeric | Months since the customer started the subscription | Lower tenure often increases churn risk |
| monthly_active_days | numeric | Active days in the last month | Lower activity can signal disengagement |
| avg_session_minutes | numeric | Average length of a product session | Very short sessions can indicate weak usage |
| support_tickets_90d | numeric | Support tickets opened in the last 90 days | High values can reflect friction or failure |
| team_size | numeric | Number of users on the account | Interacts with plan fit |
| integrations_connected | numeric | Number of external integrations configured | Strong fit for enterprise accounts |
| training_hours_last_90d | numeric | Training / enablement hours completed recently | Strong onboarding can reduce churn |
| nps_score | numeric | Net promoter score style sentiment feature | Low sentiment can combine with usage decline |
| billing_failures_6m | numeric | Failed payment events in the last 6 months | Repeated failures often raise churn risk |
| usage_drop_pct_30d | numeric | Percent drop in recent usage | Large drop can indicate disengagement |
| plan_tier | categorical | basic / pro / enterprise | Interacts with team size and integrations |
| contract_type | categorical | monthly / annual | Monthly plus short tenure increases risk |
| industry | categorical | Account industry segment | Small contextual effect |
| region | categorical | Broad geographic region | Small contextual effect |
| onboarding_quality | categorical | poor / adequate / strong onboarding assessment | Strong onboarding can reduce early churn |
| advanced_features_enabled | categorical | Whether advanced product features are turned on | Low usage plus disabled features can be risky |
| churn_risk | target | 1 = elevated churn risk, 0 = likely renewal | Prediction target |

