### Dataset: Concrete Compressive Strength

In this assignment, you will work with a modified version of the **Concrete Compressive Strength** dataset, originally published through the UCI Machine Learning Repository.

Each row represents a concrete mixture tested under controlled laboratory conditions. The input features describe the composition of the mixture (for example, cement, water, aggregates, and additives) as well as the curing time. The target variable is the measured **compressive strength of the concrete**, expressed as a continuous numeric value.

This version of the dataset has been intentionally prepared for instructional use:

- It includes both **numeric and categorical features**
- Selected features contain **missing values**, requiring imputation
- Feature scales vary substantially, making **proper scaling essential**
- The regression target includes mild noise to reflect real-world measurement variability

You may assume the dataset is appropriate for supervised learning, but **model performance is not the goal**. The purpose of this assignment is to integrate pipeline-based preprocessing with a manually implemented regression model and optimization procedure, and to observe and reason about optimization behavior rather than to achieve high predictive accuracy.

