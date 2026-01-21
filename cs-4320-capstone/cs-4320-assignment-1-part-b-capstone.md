CS 4320 Levi Dockstader -- 1/19/2026

# Part B – Week 1 Capstone Assignment: Capstone Initialization

---

## 1. Project Context (Brief)

* **Project Title:**        Electrical Grid Fault Detection
* **Data Modality:**        Tabular
* **Task Type:**            Classification
* **One-Sentence Goal:**    Using current and voltage values of a simulated electrical grid, predict when there has been an electrical fault and what type of fault has occured.

---

## 2. This Week’s Technique and Its Assumptions

* **Technique / Model Family Covered This Week:**       Not applicable at this stage
* **Key Assumptions of This Technique:** (1–2 bullets)  Not applicable at this stage

**Fit Assessment (required):**

> I expect this technique to be a **good / partial / poor** fit for my project because:

(2–4 sentences)

Not applicable at this stage

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week’s technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:**   Not applicable at this stage

* **Why this representation was reasonable for this week:** Not applicable at this stage

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
* What you intentionally did *not* attempt and why
* Any constraints encountered (data, labels, compute, time)

I've chosen to build my capstone project in a local python project using Git for version control. I've created a public Github repository accordingly. I expect to use my local system most often this semester out of convenience since I prefer it over Google Colab. Also, VS Code has better tools for debugging, better version control with Git, and has better reproducibility with a virtual environment. While Colab is great for easy setup and sharing to get started fast, I feel VS Code is better overall for a long-term project like this capstone.

This week I brainstormed ideas for this project. I ended up finding a kaggle dataset documenting simulation results of fault and normal operation values for voltage and current in an electrical grid. I decided I could use this dataset to make a ML model that predicts when there has been an electrical fault based on current and voltage values.

After researching how, I wrote a short python script that imported the dataset and printed some basic information about it. To write my script, I began with the Part A starter file and adjusted it to my needs, using online resources to learn how to import a kaggle dataset using kagglehub and how to use pathlib to navigate to where the csv is stored. To show my dataset successfully loaded, I have copied my code and the output result at the bottom of this document.

Looking at my data, I can immediately see a couple constraints on how useful it could be. First, it is built off of a simulation, which doesn't accurately account for the noise and quality of real data. Additionally, the data is already "clipped" -- it's not a transient history of the signal being measured. Since electrical faults often evolve over time, it is limiting to only look at a single-instant snapshot for each event being tracked. This will probably make working with the data more simple, but I will not be able to model fault progression, fault-indicative warning signs, or transient behavior. I will also not be able use consider signal processing methods since the original signal is not preserved in this dataset. I think it would be more interesting and insightful to work with raw voltage/current waveforms, but I am constrained by this form of data.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

Not applicable at this stage.

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

(1–2 thoughtful paragraphs)

Not applicable at this stage.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

Not applicable at this stage.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week’s technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not applicable at this stage.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: **optional unless explicitly requested**
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results

Part B code for this week:

import pandas as pd
import kagglehub
from pathlib import Path


def main():
    print("=" * 60)
    print("ML Dataset Loading Test Script")
    print("=" * 60)
    print()

    # Step 1: Load CSV with pandas
    # Download latest version
    path = kagglehub.dataset_download("esathyaprakash/electrical-fault-detection-and-classification")

    print("Path to dataset files:", path)

    # Using the path, navigate to where the csv is stored.
    dataset_dir = Path(path)

    # Find csv file -- looks for the first csv file in given directory
    csv_file = next(dataset_dir.glob("*.csv"))

    print("Loading CSV:", csv_file)

    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")

    # Step 2: Show a few lines from the file
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print()

    print("Dataset info:")
    print(df.info())
    print()

    print("=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()


(.venv) PS C:\Users\levid\School_Programming\CS4320> & C:\Users\levid\School_Programming\CS4320\.venv\Scripts\python.exe c:/Users/levid/School_Programming/CS4320/cs-4320-capstone/cs-4320-hw1-part-b.py
============================================================
ML Dataset Loading Test Script
============================================================

Path to dataset files: C:\Users\levid\.cache\kagglehub\datasets\esathyaprakash\electrical-fault-detection-and-classification\versions\2
Loading CSV: C:\Users\levid\.cache\kagglehub\datasets\esathyaprakash\electrical-fault-detection-and-classification\versions\2\classData.csv
Successfully loaded 7861 rows and 10 columns
First 5 rows of the dataset:
   G  C  B  A          Ia          Ib          Ic        Va        Vb        Vc
0  1  0  0  1 -151.291812   -9.677452   85.800162  0.400750 -0.132935 -0.267815
1  1  0  0  1 -336.186183  -76.283262   18.328897  0.312732 -0.123633 -0.189099
2  1  0  0  1 -502.891583 -174.648023  -80.924663  0.265728 -0.114301 -0.151428
3  1  0  0  1 -593.941905 -217.703359 -124.891924  0.235511 -0.104940 -0.130570
4  1  0  0  1 -643.663617 -224.159427 -132.282815  0.209537 -0.095554 -0.113983

Dataset info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7861 entries, 0 to 7860
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   G       7861 non-null   int64  
 1   C       7861 non-null   int64  
 2   B       7861 non-null   int64  
 3   A       7861 non-null   int64  
 4   Ia      7861 non-null   float64
 5   Ib      7861 non-null   float64
 6   Ic      7861 non-null   float64
 7   Va      7861 non-null   float64
 8   Vb      7861 non-null   float64
 9   Vc      7861 non-null   float64
dtypes: float64(6), int64(4)
memory usage: 614.3 KB
None

============================================================
Test completed successfully! ✓
============================================================
