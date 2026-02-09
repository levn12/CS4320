# Part B – Week 3 Capstone Assignment - Levi Dockstader

---

## 1. Project Context (Brief)

* **Project Title:**        Electrical Grid Fault Detection
* **Data Modality:**        Tabular
* **Task Type:**            Classification
* **One-Sentence Goal:**    Using current and voltage values of a simulated electrical grid, predict when there has been an electrical fault and what type of fault has occurred.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** This week was about how to preprocess data for use in a machine learning model. Specifically, this includes splitting the full dataset into training, validation, and testing datasets, and modifying the data through scaling, encoding, or other means to make the data useful in modeling without leakage.
* **Key Assumptions of This Technique:** 
Specific to the data I'm working with, I'm assuming that
    (1) The current and voltage numbers look similar across all three datasets so scaling works for all of them.
    (2) Each fault observation is independent so one row doesn't affect another.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

The features are all numbers (currents and voltages), which makes them easy to scale. The targets (G, C, B, A columns) have already been one-hot encoded, making preprocessing quite simple. The dataset is large (7,861 rows) and is since each data point is independent, there is no need for any kind of splitting strategy other than random selection. Predicting which wires have faults is a standard classification problem. There are no missing values  tricky dependencies, which means I can follow a straightforward process.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** I used the raw measurements (Ia, Ib, Ic, Va, Vb, Vc) without any fancy feature engineering.

* **Why this representation was reasonable for this week:** Current and voltage are direct measurements that are easy to understand. Scaling (standardizing) is the only prep work needed before modeling. I didn't try to invent new features because the goal was just to see if the data could be prepared effectively. Raw features (even though they're scaled) keep things simple. Since there were no missing values in my data points, no summary statistics were necessary as proxy values for missing data.

---

## 4. What Was Attempted

**What I implemented:**
In my code this week, I followed this process:

* Loaded the electrical fault dataset (7,861 rows and 10 columns)
* Checked for missing data—found zero missing values
* Split the data randomly into train (70%), validation (15%), and test (15%) using a fixed seed so it's reproducible. Quantitatively, this split my 7.861 rows into 5,503 for training, and 1,179 each for validation and training. Random split works well because each point is totally independent.
* Separated the target columns (G, C, B, A) from the features (Ia, Ib, Ic, Va, Vb, Vc)
* Scaled the features using the mean and standard deviation from training data only, then applied the same scaling to validation and test.
* Wrote the code clearly so the leakage prevention is obvious

**What I intentionally skipped:**
* I didn't investigate if some faults are rarer than others or attempt to normalize the "importance" of faults relative to their frequency in the data; I want to see if that creates a problem for modeling first.
* I didn't create new features or proxy values for my data, raw measurements are enough for now
* I didn't attempt any encoding becuase all my numbers are already numeric (no categories to convert)
* I didn't end up needing to use any imputation strategies because there were no missing values, so nothing to fill in.
* I didn't do any statistical checks for things like outliers that might need excluded. I'll deal with that if it matters during modeling

**Problems I ran into:** I didn't have any conceptual problems. The data is clean and straightforward. I always run into issues learning new syntax or finding available functions to streamline my work, but online resources and AI are very helpful for locating the best pandas or numpy function to do what I need. The overall process was very clear from the help given in the supplemental file. Additionally, the values I'm working with, the data size, and the simplicity of the operations I'm working with made implementing my code this week clean and simple; computing didn't take too much time and everything was cleanly prepped before I had to deal with it.

**Note on data leakage considerations:** 
To be very careful that no data was leaked in the preprocessing, here were a few things I considered:
* I split the data first before doing any operations that might affect the machine learning process. Breaking data into training, validation, and testing sets was the first step.
* Scaling was done from metrics calculated solely from training data. If I calculated the average and spread of numbers using the full dataset first, then the test set would secretly know about the training data. Instead, I calculate averages only from training, then use those same numbers to scale validation and test.
* I separated all of the targets from the features. The target columns (G, C, B, A) are what we want to predict, so they can't be used as input numbers. I broke the whole set of data into inputs and outputs.

---

## 5. Observations

* The data I ended with is complete when I count how much I ended with vs. how much I started with. All 7,861 rows survived the split with no data loss.
* The way I split the data seems like it will work well. The three groups (train/val/test) have correct sizes with no weird distribution issues.
* The values of different features started out on relatively different scales. Current values go from about -800 to +100, but voltage values go from about -0.3 to +0.4. Standardization fixed this by bringing both to zero mean and unit spread.
* I haven't looked at how often each fault type appears or really considered if there's an imbalance that might affect the training process of my model. That can be thought about more deeply during modeling when I know how my machine will think and learn.
* The data looks clean; I didn't identify any strange values (NaN)or corrupted rows.

---

## 6. Interpretation and Judgment

To summarize, the preparation process went smoothly with no surprises. The data is clean, there are no missing values, and the random split makes sense since each row is independent. Scaling was straightforward because all features are numbers. The biggest question I have is how the four fault possibilites affect each other, so I know what direction to aim my model towards. I know that multiple fault types can happen at once (out of G, C, B, and A), so I'm trying to think what strategy would work best to most simply predict all the possible outcomes. For example, if I use clustering, there are 2^4 = 16 different categories to consider all the different combinations of fault types. However, there are only four different lines that can have a fault. Maybe clustering isn't the best approach if there's a simpler way that thinks about each fault differently than a category.

The way I scaled features using z-scores like we talked about in class (normalizing because currents are big numbers, voltages are small) doesn't hurt anything because scaling just brings them to the same standard form without really affecting pattern built into the values. At this stage, I haven't seen any big problems that I haven't been able to deal with using methdos we've talked about in lecture. However, it's difficult right now if my assumptions about how the data behaves really will hold until I can test my data on a model. Overall, the data looks ready to train a model on. The next step is to actually try some models and see how they perform.

---

## 7. Forward-Looking Adjustment

**Before modeling:**

Answering the first question on the list, here's what I plan on keeping, changing, or discarding for next time:

* **Keep:** The way I split the data and scaled it. Both prevent leakage, are easy, and are solid.
* **Change:** Check if the four fault types are balanced and see if I can find how they correlate with each other. If some faults are rare or usually happen together, I might need to change my approach to account for that.
* **Add:** I haven't really considered trying some feature engineering like looking at mathematical relationships between current and voltage, or statistics about their variation. The raw measurements are fine for a baseline but might not be optimal. For example, maybe I can consider the really simple relationship between the two through Ohm's law (V=IR), and actually monitor resistance of the line rather than voltage and current directly.
* **Discard:** As a side note, I will likely discard some of the things we used in part A of this assignment, but I already didn't use them here in part B. For example, I did not need to implement one-hot encoding on categorical features or imputation strategies for missing data. They just aren't necessary for my dataset.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

Not applicable at this stage. The standard data prep approach works fine for this project. The data is clean and suitable for machine learning without major complications.

--- 

## Submission Notes

For completeness, I have copied my code for this week here in case it needs referenced for clarity. It closely follows the structure of the help file for part A.

```python
"""
CS 4320 — Assignment 3 (Part B)

Workflow:
1) Load data
2) Split into train/val/test with seed for reproducibility
3) Separate target y from features X
4) Fit preprocessing on TRAIN ONLY:
   - numeric mean
   - scaling mean/std
5) Apply those artifacts to val/test
"""

import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\levid\School_Programming\CS4320\cs-4320-capstone\electrical_fault_data.csv"
SEED = 4320  # Same seed as part A for convenience.

""" Target: Detect electrical fault in wires G, C, B, A and which columns fault occurred in """
TARGET_COLS = ["G", "C", "B", "A"]  # Columns representing fault detection in each wire


def split_indices(n: int, seed: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """Deterministic split using a seeded permutation (same idea as lecture)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    df = pd.read_csv(CSV_PATH)

    # Check data for missing values and print result to make sure nothing is missing.
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    print(missing_counts)

    # 1) Split - create indices for copying from original data
    train_idx, val_idx, test_idx = split_indices(len(df), SEED)

    # Create train/val/test splits by copying from original data using the indices.
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 2) Separate targets
    y_train = train_df[TARGET_COLS].to_numpy(dtype=float)
    y_val   = val_df[TARGET_COLS].to_numpy(dtype=float)
    y_test  = test_df[TARGET_COLS].to_numpy(dtype=float)

    # 3) Choose feature columns (drop targets)
    X_train = train_df.drop(columns=TARGET_COLS)
    X_val   = val_df.drop(columns=TARGET_COLS)
    X_test  = test_df.drop(columns=TARGET_COLS)

    # 4) FIT scaling on TRAIN ONLY
    # Calculate the mean and std for numeric columns from X_train.
    X_num_means = X_train.mean()
    X_num_stds = X_train.std()

    # Then apply to X_train / X_val / X_test using the formula: (X - mean) / std
    X_train = (X_train - X_num_means) / (X_num_stds)
    X_val = (X_val - X_num_means) / (X_num_stds)
    X_test = (X_test - X_num_means) / (X_num_stds)

    # No imputation is needed since there are no missing values.

    # Print shapes and other info to double check before moving on to modeling.
    print("X shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("y shapes:", y_train.shape, y_val.shape, y_test.shape)

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    print("Numeric cols:", X_train.columns.tolist())


if __name__ == "__main__":
    main()

```