# Capstone Postmortem
Levi Dockstader
CS 4320 - Spring 2026

## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection Postmortem
* **Data Modality:** Tabular
* **Task Type:** Primarily classification, with one unsupervised structure-discovery phase
* **One-Sentence Goal:** Use measured phase currents and voltages from a simulated three-phase electrical system to detect whether a fault is present and, when possible, classify the exact fault pattern.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Capstone retrospective across multiple machine learning approaches: preprocessing and leakage-safe evaluation, logistic regression, threshold tuning, regularization, Naive Bayes, k-Nearest Neighbors, SVMs, decision trees, ensembles, PCA + k-means, multilayer perceptrons, and a final engineered-feature Random Forest workflow.
* **Key Assumptions of This Technique:**
  * The six measured signals `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc` contain enough information to separate normal operation from faulted operation, and possibly to separate different fault patterns.
  * Rows are independent enough that random stratified train/validation/test splits are meaningful.
  * The observed label structure can be studied in two ways: a simpler binary `any fault` task and a harder multiclass fault-pattern task.
  * Physically motivated proxy features can make the fault structure easier for some models to use than the raw measurements alone.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

This capstone ended up being a really good example of why it helps to compare both model families and feature representations instead of committing too early. Early in the semester I mostly used the six raw electrical measurements and a simpler binary target, which was enough to learn a lot about optimization and evaluation. By the end, though, the more important lesson was that the feature representation changed the story just as much as the model family did.

The biggest shift was that I stopped thinking only in terms of "which model wins on raw data?" and started asking "which model makes the most sense once I represent the electrical system in a more meaningful way?" On the raw multiclass task, kNN was strongest. But after I built impedance- and phase-related proxy features and reran the comparison, Random Forest became the clearest final choice. That made this retrospective a much better fit than any single weekly method by itself.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** The final representation uses both the six measured electrical features `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc` and a smaller set of engineered proxy features:
  * `z_proxy_A`, `z_proxy_B`, `z_proxy_C`
  * `z_proxy_3ph`
  * `current_vector_mag`, `voltage_vector_mag`
  * `abs_current_sum_signed`
  * `current_unbalance`, `voltage_unbalance`
  * `abs_instantaneous_power_proxy`

* **Why this representation was reasonable for this week:**

The raw measurements were still the starting point all semester, and they are still the most honest inputs because they are closest to what a real sensor would provide. I intentionally did **not** use the fault-indicator columns `G`, `C`, `B`, and `A` as model inputs, because those columns define the label and would leak the answer directly into the model.

The main thing that changed in the final folder is that I stopped using only raw features and started using a small set of physically motivated proxies built from them. I call them proxies on purpose because they are not exact power-systems calculations from a full phasor model. They are simpler approximations based on the snapshot data I actually have.

Those proxy features made sense physically for a few reasons:

* Faults often change the relationship between voltage and current, so `|V| / |I|` style features act like rough impedance clues.
* Faults can also make the three phases less balanced, so current and voltage unbalance features are helpful.
* Vector magnitudes summarize how "large" the three-phase current and voltage state is overall.
* The signed current sum and power-style proxy help capture how the phases line up or cancel out, which is hard to see from one raw column at a time.

This ended up mattering a lot. On the raw multiclass task, the data looked more like a geometry problem where kNN could use local neighborhoods well. After I added the proxy features, the same data looked much friendlier to tree-based rules, because a Random Forest could split directly on things like imbalance, impedance-like ratios, and combined magnitudes.

I still used two target representations during the semester:

* **Binary proxy target:** `fault` vs `no fault`, where any active indicator counts as a fault
* **Multiclass target:** one label formed by concatenating the four fault bits into patterns such as `0000`, `0110`, `0111`, `1001`, `1011`, and `1111`

The binary target was useful early because it let me test training behavior and basic separability. The multiclass target turned out to be much more important for the real project because it exposed actual differences between models instead of making the task almost too easy.

---

## 4. What Was Attempted

This capstone developed in stages rather than through one single model. The overall system design in the final folder is:

* Load `electrical_fault_data.csv`
* Build the multiclass fault-pattern label from the four fault bits
* Build raw, proxy-only, and raw-plus-proxy feature sets
* Split into train / validation / test with reproducible seeds
* Tune on validation first and keep test untouched until the end
* Save outputs that make the final run easy to inspect and rerun

The main methods I explored across the semester were:

* **Preprocessing and leakage control:** early work focused on correct splitting, scaling, and keeping the fault-bit columns out of the feature matrix.
* **Logistic regression baseline:** this gave me a simple linear model to understand loss behavior and establish a baseline.
* **Threshold tuning for binary classification:** this showed that model quality is not just about the classifier itself, but also about how the operating threshold changes false positives and false negatives.
* **Regularization sweeps:** these helped check whether poor performance came from overfitting or from representational limits. For the logistic setup, regularization usually did not change much.
* **Gaussian Naive Bayes vs kNN:** this showed that local neighborhood structure mattered more than Gaussian independence assumptions.
* **Linear SVM vs RBF SVM:** this was one of the clearest signs that nonlinear boundaries fit the binary fault task much better than linear ones.
* **Decision tree, Random Forest, and Gradient Boosting:** these explored nonlinear rule-based models on the harder multiclass task.
* **PCA and k-means clustering:** this was used to study the geometry of the measurement space and compare unsupervised structure to known fault labels after the fact.
* **MLPs / neural networks:** these tested whether learned nonlinear feature combinations could outperform simpler methods on multiclass prediction.
* **Raw multiclass comparison:** I directly compared kNN, RBF SVM, Random Forest, and MLP on the same split using raw features only.
* **Proxy experiments:** after that comparison, I ran extra experiments in `capstone-experiments` to compare raw features against proxy-enhanced features and to compare Random Forest against MLP on the proxy representation.
* **Final capstone model:** I settled on a tuned Random Forest using the combined raw-plus-proxy feature set and packaged the whole workflow in `capstone-final/run_capstone_final_model.py`.

That script:

* loads the dataset
* builds the multiclass labels
* builds the proxy features
* benchmarks `raw_only`, `proxy_only`, and `raw_plus_proxy`
* tunes the final Random Forest
* evaluates once on the held-out test split
* saves outputs to:
  * `cs-4320-capstone/capstone-final/outputs/plots`
  * `cs-4320-capstone/capstone-final/outputs/tables`
  * `cs-4320-capstone/capstone-final/outputs/model`

The final outputs are:

* `outputs/run_summary.md`
* `outputs/tables/final_metrics.csv`
* `outputs/tables/feature_set_screening_summary.csv`
* `outputs/tables/feature_importance.csv`
* `outputs/tables/classification_report.csv`
* `outputs/tables/test_predictions.csv`
* `outputs/plots/test_confusion_matrix_normalized.png`
* `outputs/model/final_random_forest_pipeline.pkl`

Training setup and constraints stayed fairly consistent:

* Dataset size was `7,861` rows
* Known fault patterns observed in the data were `0000`, `0110`, `0111`, `1001`, `1011`, and `1111`
* The final split sizes were `4,716` train, `1,572` validation, and `1,573` test
* I kept preprocessing leakage-safe by fitting imputation and scaling on training data only when needed
* I used validation performance for model selection and saved test evaluation for the end
* The data is simulated, which means strong performance should still be interpreted carefully because a real grid would likely have more noise and drift

What I intentionally did not attempt:

* I did not treat the project as a time-series problem because the dataset is row-based tabular data rather than a sequential waveform dataset.
* I did not use the fault-indicator bits as inputs.
* I did not do massive hyperparameter searches, because the capstone goal was to learn from method behavior rather than chase tiny gains.

---

## 5. Results or Observations

Several results stood out across the semester, and the final folder changed the conclusion in an important way.

### Dataset and raw-structure observations

* The dataset had `7,861` rows and six raw measurement features.
* The fault-pattern distribution was fairly balanced across the six observed classes.
* PCA on the raw features showed strong low-dimensional structure:
  * `PC1 = 30.74%`
  * `PC2 = 25.70%`
  * first two PCs combined = `56.44%`
  * first four PCs combined = `97.21%`

That mattered because it showed the electrical measurements are highly structured rather than noisy or random.

### Binary-task observations

The binary `any fault` task was useful early on, but it often turned out to be easier than the full project goal.

* **Logistic regression baseline:** around `0.6992` accuracy, `1.0000` recall, and `0.8229` F1 in one early setup
* **Threshold-tuned logistic regression:** one thresholding policy produced only `0.5340` accuracy and `0.5391` balanced accuracy on test
* **Regularization study:** tuning `C` and related settings barely changed performance, which suggested model capacity control was not the main bottleneck for the logistic setup
* **GaussianNB:** validation F1 around `0.9810`
* **kNN on the binary task:** essentially perfect, with final test accuracy/F1 of `1.0000`
* **RBF SVM on the binary task:** nearly perfect, with final test accuracy `0.9968`, balanced accuracy `0.9977`, and F1 `0.9977`

The pattern here was pretty clear: once I allowed nonlinear or local-decision methods, binary fault detection became almost too easy.

### Multiclass-task observations on raw features

The multiclass fault-pattern task was much more informative because it exposed real differences between methods.

* **Decision Tree:** test accuracy `0.8798`, balanced accuracy `0.8590`, macro F1 `0.8585`
* **Earlier multiclass MLP run:** test accuracy `0.8525`, macro F1 `0.8225`
* **Later deep-learning assignment:** validation accuracy `0.8592`, test accuracy `0.8567`, macro F1 `0.7855`

The raw-feature side-by-side comparison in `capstone-experiments/multiclass_fault_comparison_outputs` gave this final snapshot:

* **kNN:** accuracy `0.8887`, balanced accuracy `0.8691`, macro F1 `0.8691`
* **Random Forest:** accuracy `0.8792`, balanced accuracy `0.8579`, macro F1 `0.8575`
* **RBF SVM:** accuracy `0.8538`, balanced accuracy `0.8275`, macro F1 `0.8254`
* **MLP:** accuracy `0.8449`, balanced accuracy `0.8172`, macro F1 `0.8143`

If I had stopped there, I would have picked **kNN**. That was the right choice for the raw-only comparison.

### Proxy-experiment observations

The final folder exists because I did **not** stop there. After seeing the raw-feature comparison, I ran more experiments focused on physically motivated proxies.

The first big result was that proxy features changed the problem dramatically for Random Forest:

* In `capstone-experiments/proxy_rf_mlp_experiment/outputs/comparison_summary.md`, **Random Forest with raw-plus-best-proxy features** reached test balanced accuracy `0.9971` and test F1 `0.9971`
* The same experiment showed **Random Forest with raw-only features** at test balanced accuracy `0.8594` and test F1 `0.8590`
* That is a huge jump, and it told me the representation was the real missing piece.

The second big result was that Random Forest benefited from the proxy representation much more than MLP did:

* In `selected_proxy_outputs/comparison_summary.md`, **Random Forest** reached test balanced accuracy `0.9941` and test F1 `0.9941`
* The matched **MLP** on that proxy set reached test balanced accuracy `0.8500` and test F1 `0.8295`

That made the final direction much clearer: once I gave the model features that reflected impedance-like behavior and phase imbalance, the tree ensemble used them extremely well.

### Final model observations

The final benchmark saved in `capstone-final/outputs/tables/feature_set_screening_summary.csv` was:

* `raw_plus_proxy`: validation balanced accuracy `0.9971`
* `proxy_only`: validation balanced accuracy `0.9971`
* `raw_only`: validation balanced accuracy `0.8449`

So the final folder confirmed that proxies were doing most of the heavy lifting, and that raw-only performance was much weaker for the final Random Forest workflow.

The final run in `capstone-final/outputs/run_summary.md` and `final_metrics.csv` gave:

* **Model family:** `RandomForestClassifier`
* **Final feature representation:** `raw_plus_proxy`
* **Best hyperparameters:** `n_estimators=300`, `max_depth=None`, `max_features="sqrt"`, `min_samples_leaf=1`, `min_samples_split=2`, `class_weight="balanced_subsample"`
* **Validation balanced accuracy:** `0.9971`
* **Test accuracy:** `0.9968`
* **Test balanced accuracy:** `0.9963`
* **Test macro F1:** `0.9963`
* **Out-of-bag score:** `0.9981`

The held-out test confusion matrix was especially interesting because it shows how small the remaining error really is:

* `0000`, `0110`, `0111`, `1001`, and `1011` were all classified perfectly on test
* The only test mistakes were `5` rows from class `1111`
* Those `5` rows were predicted as `0111` (`2` rows) or `1011` (`3` rows)

That means the old "hard class" pattern never fully disappeared, but the final model reduced it to a very small leftover issue instead of a major weakness.

The top features in `feature_importance.csv` were also almost all proxy features:

* `abs_current_sum_signed`: `0.3199`
* `z_proxy_3ph`: `0.0974`
* `current_vector_mag`: `0.0907`
* `z_proxy_A`: `0.0822`
* `voltage_vector_mag`: `0.0786`
* `current_unbalance`: `0.0637`
* `z_proxy_B`: `0.0584`
* `z_proxy_C`: `0.0389`
* `abs_instantaneous_power_proxy`: `0.0360`

That lines up very well with the physical motivation for using proxies in the first place.

### Unsupervised observations

The unsupervised work was still valuable, but in a different way from the final classifier.

* Best tested k-means setting by silhouette score was `k=8`
* Silhouette score at `k=8` was `0.4006`
* Post hoc adjusted Rand index at `k=8` was only `0.1688`

That told me the measurement space definitely has structure, but the natural geometric groupings do not line up perfectly with the labeled fault patterns.

---

## 6. Interpretation and Judgment

The biggest lesson from this capstone is that **representation mattered at least as much as model family**. Earlier in the semester I kept asking which model was best, but the final folder made it obvious that this was not the full question. On raw multiclass features, kNN really was the best of the models I compared. But once I moved to a more physically meaningful representation, Random Forest became the strongest model by a wide margin.

That shift makes sense to me now. On raw features, the class structure seems to behave a lot like a neighborhood problem. kNN can do well there because nearby examples in the six-dimensional space tend to share the same label. But the raw values do not directly expose the electrical relationships I actually care about. A Random Forest can only split on what it sees, so if the important signal is hidden inside ratios, imbalance, or combined magnitudes, the forest is at a disadvantage.

The proxy features fixed that. Impedance-style ratios matter here because a fault changes how voltage and current relate. Unbalance matters because a fault often disturbs the symmetry across phases. Magnitude and power-style summaries matter because they capture the overall "size" and alignment of the electrical state. Once I encoded those ideas into the feature set, the Random Forest had much better variables to split on, and its performance jumped from the mid-0.85 range to basically perfect on the multiclass task.

This also helped explain some earlier results that confused me. In the homework tree/ensemble comparison, I was surprised that a single tree slightly beat Random Forest. Looking back, I do not think that meant Random Forest was a bad fit. I think it mostly meant I was still asking the model to work from a weaker representation. The later proxy experiments gave a much more convincing picture: Random Forest was actually a very good fit, but it needed the right features.

I also learned that more complex models did **not** automatically do better. The MLP was reasonable, and it definitely learned real signal, but it never became the strongest method on this tabular problem. Even with proxy features, the Random Forest beat the MLP clearly. That fits a common pattern for smaller tabular datasets: a well-chosen tree ensemble plus good features can be hard to beat.

Overall, I think the final Random Forest was the right choice because it balanced several things at once:

* it used the harder and more meaningful multiclass target
* it benefited strongly from physically motivated proxy features
* it performed much better than the raw-feature version
* it saved a clean rerunnable pipeline and readable outputs in the final folder

I also think this project has a pretty clear real-world use, even if the current version is still more of a strong capstone prototype than a deployment-ready system. A model like this could help with faster fault screening, operator decision support, and automatic labeling of likely fault patterns from incoming sensor data. In practice, that could mean helping utilities detect abnormal conditions faster, narrow down which kind of fault is likely happening, and reduce the amount of manual inspection needed after an event. Even a model that is not trusted to make the final decision could still be useful if it helps prioritize alarms or point engineers toward the most likely problem first.

The part that seems most useful to me is that the final model is not just saying "something is wrong." It is trying to separate specific fault patterns. That could make it more helpful for troubleshooting, because different fault patterns can suggest different physical problems or different next steps. A classifier like this could be used as an early triage layer inside a larger monitoring system, where its job is to quickly score incoming readings and flag the most likely fault class for review.

---

## 7. Forward-Looking Adjustment

If I continued this project, I would keep the same leakage-safe evaluation discipline and the proxy-based Random Forest workflow, but I would push the next phase in a few specific ways:

* Keep the multiclass fault-pattern framing as the main task, because it is much more informative than the binary proxy.
* Stress-test the proxy features under noisier or shifted data, since the current dataset is simulated and very clean.
* Focus error analysis on the small remaining `1111` confusion cases to understand whether those rows are physically borderline.
* Try a few more phase-based proxy ideas, especially if I can get more realistic electrical context or time-windowed data.
* If richer data were available, move from single-row snapshots toward a more realistic temporal setup.
* Add deployment-oriented checks such as latency, confidence thresholds, fallback rules, and monitoring for data drift.

If I had more time, I would also like to compare the final Random Forest against a gradient-boosted tree model on the same proxy representation, just to see whether the representation or the ensemble type is doing more of the work.

For actual deployment, I think the current model would need a few important upgrades before it could be trusted in the real world:

* It would need validation on real sensor data rather than only simulated data.
* It would need testing under noise, missing values, bad sensor readings, and changing operating conditions.
* It would probably need to run on a rolling time window instead of single independent rows, because real grid events develop over time.
* It would need a confidence or abstain policy so that uncertain predictions can be handed off to a human operator or a more conservative backup rule.
* It would need monitoring after deployment, because drift in sensor behavior or system configuration could slowly reduce accuracy.

I also do not think this kind of model should directly trip protective equipment on its own in its current form. A more realistic use would be as a support tool inside a larger fault-management pipeline. For example, it could rank likely fault types, help filter false alarms, highlight suspicious feeder states, or help label historical events for engineers. If the goal were fully automated control, I would want much more evidence on reliability, edge cases, and failure modes first.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

Several methods were only partial fits, and those mismatches were actually some of the most useful parts of the project.

* **Logistic regression** was a helpful baseline, but it was too limited for the real structure of the task.
* **Threshold tuning** was worth doing, but it also showed that no threshold can rescue a weak underlying model.
* **Naive Bayes** needed adaptation because count-based assumptions did not match continuous electrical features.
* **k-means clustering** found real geometric structure, but that structure did not align closely with the supervised labels.
* **Early raw-feature ensemble results** made Random Forest look weaker than I expected, but later proxy experiments showed that the representation was the bigger issue.
* **MLPs** learned useful nonlinear structure, but they never became the best option for this particular tabular dataset.

The biggest machine-learning lesson from those mismatches is that underperformance does not always mean "wrong model." Sometimes it means the target is too easy, the representation is too weak, or the model is being asked to solve the problem in a less natural coordinate system. This capstone was especially good for learning that, because the final answer changed only after I changed the representation in a way that made more physical sense.


## Final Code:
```python
"""
Final capstone model for multiclass electrical fault classification.

This script keeps the final workflow in one place:
1. Load the electrical fault dataset.
2. Build raw features and a smaller set of impedance / phase-related proxy features.
3. Make a smaller set of high-value exploratory plots for both raw and proxy features.
4. Compare raw-only, proxy-only, and raw-plus-proxy feature sets.
5. Tune a final Random Forest on the best representation.
6. Train the final model and save metrics, plots, and model artifacts.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")


# Basic file locations for the final capstone package.
# The script lives in `capstone-final`, while the CSV lives one folder up
# in the main capstone directory.
BASE_DIR = Path(__file__).resolve().parent
CAPSTONE_DIR = BASE_DIR.parent
DATA_PATH = CAPSTONE_DIR / "electrical_fault_data.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

# Fault columns define the target label.
# Raw feature columns are the measured sensor values used as direct inputs.
FAULT_COLS = ["G", "C", "B", "A"]
CURRENT_COLS = ["Ia", "Ib", "Ic"]
VOLTAGE_COLS = ["Va", "Vb", "Vc"]
RAW_FEATURE_COLS = CURRENT_COLS + VOLTAGE_COLS
FEATURE_SET_ORDER = ["raw_only", "proxy_only", "raw_plus_proxy"]

# Final proxy set: keep it small and interpretable.
# These were selected to summarize impedance-like behavior, magnitude, and imbalance.
PROXY_FEATURE_COLS = [
    "z_proxy_A",
    "z_proxy_B",
    "z_proxy_C",
    "z_proxy_3ph",
    "current_vector_mag",
    "voltage_vector_mag",
    "abs_current_sum_signed",
    "current_unbalance",
    "voltage_unbalance",
    "abs_instantaneous_power_proxy",
]

RANDOM_STATE = 4320
PRIMARY_METRIC = "balanced_accuracy"
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN_VAL = 0.25
EPSILON = 1e-6
MAX_SCATTER_POINTS = 3000

SCREENING_GRID = [
    {
        "n_estimators": 250,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 400,
        "max_depth": 20,
        "max_features": 0.6,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "max_features": 0.6,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "class_weight": "balanced_subsample",
    },
]

FINAL_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 22],
    "max_features": ["sqrt", 0.6],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2],
    "class_weight": ["balanced_subsample"],
}


def parse_args():
    # The final script is intentionally simple.
    # The only configurable input is the output directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def set_seed(seed):
    # Reproducibility matters for the report and for regenerating the same outputs.
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    # Create output folders if they are missing.
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_json_safe(value):
    # Convert NumPy and Path objects into regular Python / JSON-friendly values.
    if isinstance(value, dict):
        return {str(key): make_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(data, path):
    # Save metadata-style outputs as readable JSON.
    path.write_text(json.dumps(make_json_safe(data), indent=2), encoding="utf-8")


def save_dataframe(df, path):
    # Save tabular outputs as CSV.
    df.to_csv(path, index=False)


def save_text(text, path):
    # Save the markdown summary.
    path.write_text(text, encoding="utf-8")


def display_depth(value):
    # Pandas stores None in numeric columns as NaN.
    # Convert that back into a cleaner display value for saved result tables.
    return "None" if pd.isna(value) else int(value)


def clean_depth_column(df):
    # Apply the depth cleanup only if the table actually has a max_depth column.
    cleaned = df.copy()
    if "max_depth" in cleaned.columns:
        cleaned["max_depth"] = cleaned["max_depth"].apply(display_depth)
    return cleaned


def safe_ratio(numerator, denominator):
    # Several engineered features are ratios.
    # A small epsilon avoids divide-by-zero problems.
    denominator = np.maximum(np.asarray(denominator), EPSILON)
    return np.asarray(numerator) / denominator


def load_dataset():
    # Read the fixed capstone dataset and build the multiclass target label.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Turn the four fault bits into one multiclass string label, such as "0000" or "1011".
    df["fault_pattern"] = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1)

    # Only the six measured signals are used as model inputs.
    # The fault bits are label information and should not be included as features.
    raw_features = df[RAW_FEATURE_COLS].copy()
    labels = df["fault_pattern"].to_numpy()
    return raw_features, labels


def build_proxy_features(raw_features):
    # Build a smaller, easier-to-explain set of engineered features from the raw signals.
    # These are physically motivated proxies, not exact power-system calculations.
    abs_currents = raw_features[CURRENT_COLS].abs()
    abs_voltages = raw_features[VOLTAGE_COLS].abs()

    # Vector magnitudes summarize the overall size of the three-phase current
    # and voltage vectors on each row.
    current_matrix = raw_features[CURRENT_COLS].to_numpy()
    voltage_matrix = raw_features[VOLTAGE_COLS].to_numpy()
    current_mag = np.linalg.norm(current_matrix, axis=1)
    voltage_mag = np.linalg.norm(voltage_matrix, axis=1)

    # Dot product gives a simple power-like proxy.
    dot_product = np.sum(current_matrix * voltage_matrix, axis=1)
    current_sum = raw_features[CURRENT_COLS].sum(axis=1)

    proxy = pd.DataFrame(
        {
            # Phase-by-phase impedance-style proxies using |V| / |I|.
            "z_proxy_A": safe_ratio(abs_voltages["Va"], abs_currents["Ia"]),
            "z_proxy_B": safe_ratio(abs_voltages["Vb"], abs_currents["Ib"]),
            "z_proxy_C": safe_ratio(abs_voltages["Vc"], abs_currents["Ic"]),

            # Three-phase magnitude ratio.
            "z_proxy_3ph": safe_ratio(voltage_mag, current_mag),

            # Overall size summaries.
            "current_vector_mag": current_mag,
            "voltage_vector_mag": voltage_mag,

            # Absolute signed-current sum can reflect phase cancellation behavior.
            "abs_current_sum_signed": current_sum.abs(),

            # Unbalance features measure how uneven the three phases are.
            "current_unbalance": safe_ratio(abs_currents.std(axis=1, ddof=0), abs_currents.mean(axis=1)),
            "voltage_unbalance": safe_ratio(abs_voltages.std(axis=1, ddof=0), abs_voltages.mean(axis=1)),

            # Absolute power-style proxy.
            "abs_instantaneous_power_proxy": np.abs(dot_product),
        },
        index=raw_features.index,
    )

    return proxy[PROXY_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)


def build_feature_sets(raw_features, proxy_features):
    # Compare three representations:
    # 1. raw_only
    # 2. proxy_only
    # 3. raw_plus_proxy
    raw_plus_proxy = pd.concat([raw_features, proxy_features], axis=1)
    return {"raw_only": raw_features, "proxy_only": proxy_features, "raw_plus_proxy": raw_plus_proxy}


def make_model_pipeline(params):
    # Random Forest does not need scaling, but we still include median imputation
    # so the pipeline is complete and easy to reuse.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    max_features=params["max_features"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_samples_split=params["min_samples_split"],
                    class_weight=params["class_weight"],
                    bootstrap=True,
                    oob_score=True,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true, y_pred):
    # Use macro metrics so each fault class contributes equally.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_model(model, X, y):
    # Shared wrapper for model evaluation.
    return evaluate_predictions(y, model.predict(X))


def score_parameter_grid(X_train, y_train, X_val, y_val, param_grid):
    # Try every parameter setting in the grid and store the results in one table.
    if isinstance(param_grid, list):
        parameter_sets = param_grid
    else:
        parameter_sets = list(ParameterGrid(param_grid))

    rows = []
    for params in parameter_sets:
        # Train using only the training split.
        model = make_model_pipeline(params)
        model.fit(X_train, y_train)

        # Evaluate on both training and validation to watch for overfitting.
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        oob_score = float(model.named_steps["model"].oob_score_)

        rows.append(
            {
                **params,
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "oob_score": oob_score,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_precision_macro": val_metrics["precision"],
                "val_recall_macro": val_metrics["recall"],
                "val_f1_macro": val_metrics["f1"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        # Balanced accuracy is the main metric, with F1 and accuracy as tie-breakers.
        by=["val_balanced_accuracy", "val_f1_macro", "val_accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_forest_structure(forest):
    # Summarize the actual fitted tree sizes in the final forest.
    depths = [tree.tree_.max_depth for tree in forest.estimators_]
    leaves = [tree.tree_.n_leaves for tree in forest.estimators_]
    return {
        "n_trees": len(depths),
        "tree_depth_min": int(np.min(depths)),
        "tree_depth_median": float(np.median(depths)),
        "tree_depth_mean": float(np.mean(depths)),
        "tree_depth_max": int(np.max(depths)),
        "leaf_count_min": int(np.min(leaves)),
        "leaf_count_median": float(np.median(leaves)),
        "leaf_count_mean": float(np.mean(leaves)),
        "leaf_count_max": int(np.max(leaves)),
    }


def compute_confusion_df(y_true, y_pred, class_names, normalize=False):
    # Return the confusion matrix as a labeled DataFrame so it is easy to save and plot.
    mode = "true" if normalize else None
    matrix = confusion_matrix(y_true, y_pred, labels=class_names, normalize=mode)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in class_names],
        columns=[f"pred_{label}" for label in class_names],
    )


def sample_for_scatter(feature_frame, labels):
    # PCA scatterplots can get overcrowded, so cap them at a manageable size.
    if len(feature_frame) <= MAX_SCATTER_POINTS:
        return feature_frame.copy(), labels.copy()

    sampled_index = feature_frame.sample(n=MAX_SCATTER_POINTS, random_state=RANDOM_STATE).index
    sampled_labels = labels[pd.Index(feature_frame.index).get_indexer(sampled_index)]
    return feature_frame.loc[sampled_index].copy(), sampled_labels


def build_color_map(labels):
    # Use a stable color per class label within each plot.
    cmap = plt.get_cmap("tab10")
    return {label: cmap(index % 10) for index, label in enumerate(labels)}


def plot_class_balance(labels, path):
    # Plot the number of examples in each multiclass label.
    counts = pd.Series(labels, name="fault_pattern").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(counts.index, counts.values, color="tab:blue")
    ax.set_title("Class distribution by fault pattern")
    ax.set_xlabel("Fault pattern")
    ax.set_ylabel("Rows")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return counts


def plot_pca(feature_frame, labels, scatter_2d_path, scatter_3d_path, variance_path, title_prefix):
    # Standardize first so PCA is driven by structure, not by raw scale differences.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)

    # First PCA figure: class scatter in the first two principal components.
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca_2d.fit_transform(scaled)
    scatter_df = pd.DataFrame(coordinates, columns=["PC1", "PC2"], index=feature_frame.index)
    scatter_df, sampled_labels = sample_for_scatter(scatter_df, labels)
    unique_labels = sorted(pd.Series(sampled_labels).unique().tolist())
    colors = build_color_map(unique_labels)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for label in unique_labels:
        mask = sampled_labels == label
        ax.scatter(
            scatter_df.loc[mask, "PC1"],
            scatter_df.loc[mask, "PC2"],
            s=18,
            alpha=0.65,
            color=colors[label],
            label=label,
        )

    ax.set_title(f"{title_prefix} PCA scatter by fault pattern")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fault pattern", ncol=2, fontsize=9)
    fig.savefig(scatter_2d_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Second PCA figure: 3D scatter in the first three principal components.
    pca_3d = PCA(n_components=min(3, len(feature_frame.columns)), random_state=RANDOM_STATE)
    coords_3d = pca_3d.fit_transform(scaled)
    scatter_3d_df = pd.DataFrame(coords_3d, columns=[f"PC{i}" for i in range(1, pca_3d.n_components_ + 1)], index=feature_frame.index)
    scatter_3d_df, sampled_labels_3d = sample_for_scatter(scatter_3d_df, labels)

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for label in unique_labels:
        mask = sampled_labels_3d == label
        ax.scatter(
            scatter_3d_df.loc[mask, "PC1"],
            scatter_3d_df.loc[mask, "PC2"],
            scatter_3d_df.loc[mask, "PC3"],
            s=16,
            alpha=0.60,
            color=colors[label],
            label=label,
        )

    ax.set_title(f"{title_prefix} PCA 3D scatter by fault pattern")
    ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2] * 100:.2f}% variance)")
    ax.legend(title="Fault pattern", ncol=2, fontsize=8)
    fig.savefig(scatter_3d_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Third PCA figure: explained-variance view.
    pca_full = PCA(n_components=min(len(feature_frame.columns), 10), random_state=RANDOM_STATE)
    pca_full.fit(scaled)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    component_numbers = np.arange(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.bar(component_numbers, explained, alpha=0.80, label="Individual")
    ax.plot(component_numbers, cumulative, marker="o", color="black", label="Cumulative")
    ax.set_title(f"{title_prefix} PCA explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(component_numbers)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(variance_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "pc1_variance_ratio": float(pca_2d.explained_variance_ratio_[0]),
        "pc2_variance_ratio": float(pca_2d.explained_variance_ratio_[1]),
        "cumulative_variance_first_2": float(np.sum(pca_2d.explained_variance_ratio_)),
        "cumulative_variance_first_5": float(np.sum(explained[: min(5, len(explained))])),
    }


def plot_feature_set_benchmark(summary_df, path):
    # Compare the best validation result for each feature representation.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(
        summary_df["feature_set"],
        summary_df["best_val_balanced_accuracy"],
        color=["tab:blue", "tab:orange", "tab:green"][: len(summary_df)],
    )
    ax.set_title("Validation balanced accuracy by feature set")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    for index, value in enumerate(summary_df["best_val_balanced_accuracy"]):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_validation_results(results_df, path, top_k=10):
    # Plot the best hyperparameter settings from the final tuning grid.
    top_df = results_df.head(top_k).copy()
    top_df["label"] = [
        (
            f"n={row['n_estimators']}, depth={display_depth(row['max_depth'])}, "
            f"feat={row['max_features']}, leaf={row['min_samples_leaf']}, split={row['min_samples_split']}"
        )
        for _, row in top_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.barh(top_df["label"][::-1], top_df["val_balanced_accuracy"][::-1], color="tab:green")
    ax.set_title("Top Random Forest validation settings")
    ax.set_xlabel("Validation balanced accuracy")
    ax.set_ylabel("Hyperparameter setting")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(confusion_df, title, path, fmt):
    # Make a heatmap with values written inside each confusion-matrix cell.
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(confusion_df.to_numpy(), cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_xticklabels([label.replace("pred_", "") for label in confusion_df.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_yticklabels([label.replace("true_", "") for label in confusion_df.index])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    values = confusion_df.to_numpy()
    threshold = float(values.max()) / 2 if values.size else 0.0
    for row in range(confusion_df.shape[0]):
        for col in range(confusion_df.shape[1]):
            value = confusion_df.iat[row, col]
            label = f"{value:.2f}" if fmt == ".2f" else f"{int(value)}"
            color = "white" if float(value) > threshold else "black"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(feature_importance_df, path, top_k=20):
    # Show which raw and proxy features mattered most to the final forest.
    top_df = feature_importance_df.head(top_k).copy()
    colors = ["tab:blue" if group == "raw" else "tab:orange" for group in top_df["feature_group"]]

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(top_df["feature"][::-1], top_df["importance"][::-1], color=colors[::-1])
    ax.set_title(f"Top {top_k} Random Forest feature importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_feature_manifest(raw_features, proxy_features):
    # Track whether each final input column came from the raw set or proxy set.
    rows = []
    for feature in raw_features.columns:
        rows.append({"feature": feature, "feature_group": "raw"})
    for feature in proxy_features.columns:
        rows.append({"feature": feature, "feature_group": "proxy"})
    return pd.DataFrame(rows)


def build_run_summary(data_path, class_counts, screening_summary_df, best_params, forest_structure, train_metrics, test_metrics, oob_score, top_features_df):
    # Build a short markdown summary that can be read without opening all CSVs and plots.
    feature_lines = []
    for _, row in screening_summary_df.iterrows():
        feature_lines.append(
            f"- `{row['feature_set']}`: validation balanced accuracy `{row['best_val_balanced_accuracy']:.4f}`"
        )

    top_feature_lines = []
    for _, row in top_features_df.head(10).iterrows():
        top_feature_lines.append(f"- `{row['feature']}` ({row['feature_group']}): `{row['importance']:.4f}`")

    return f"""# Capstone final run summary

## Dataset
- Source CSV: `{data_path}`
- Rows: `{int(class_counts.sum())}`
- Classes: `{len(class_counts)}`
- Fault pattern counts: {", ".join([f"`{label}`={count}" for label, count in class_counts.items()])}

## Feature-set benchmark
{chr(10).join(feature_lines)}

## Final model
- Model family: `RandomForestClassifier`
- Final feature representation: `raw_plus_proxy`
- Best hyperparameters: `{json.dumps(make_json_safe(best_params), sort_keys=True)}`
- Out-of-bag score on the training split: `{oob_score:.4f}`
- Actual fitted tree depths: min `{forest_structure['tree_depth_min']}`, median `{forest_structure['tree_depth_median']:.1f}`, mean `{forest_structure['tree_depth_mean']:.2f}`, max `{forest_structure['tree_depth_max']}`
- Actual fitted leaf counts: min `{forest_structure['leaf_count_min']}`, median `{forest_structure['leaf_count_median']:.1f}`, mean `{forest_structure['leaf_count_mean']:.2f}`, max `{forest_structure['leaf_count_max']}`

## Metrics
- Train metrics: accuracy `{train_metrics['accuracy']:.4f}`, balanced accuracy `{train_metrics['balanced_accuracy']:.4f}`, macro F1 `{train_metrics['f1']:.4f}`
- Test metrics: accuracy `{test_metrics['accuracy']:.4f}`, balanced accuracy `{test_metrics['balanced_accuracy']:.4f}`, macro precision `{test_metrics['precision']:.4f}`, macro recall `{test_metrics['recall']:.4f}`, macro F1 `{test_metrics['f1']:.4f}`

## Most important features
{chr(10).join(top_feature_lines)}
"""


def main():
    # 1. Parse settings and prepare the output folders.
    args = parse_args()
    set_seed(RANDOM_STATE)

    output_dir = ensure_dir(args.output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    tables_dir = ensure_dir(output_dir / "tables")
    model_dir = ensure_dir(output_dir / "model")

    # 2. Load the dataset and build the three feature representations.
    raw_features, labels = load_dataset()
    proxy_features = build_proxy_features(raw_features)
    feature_sets = build_feature_sets(raw_features, proxy_features)
    class_names = sorted(pd.Series(labels).unique().tolist())

    # 3. Save the main dataset-level outputs: class distribution and PCA views.
    class_counts = plot_class_balance(labels, plots_dir / "class_balance.png")
    save_dataframe(class_counts.rename_axis("fault_pattern").reset_index(name="count"), tables_dir / "class_distribution.csv")

    raw_pca_summary = plot_pca(
        raw_features,
        labels,
        plots_dir / "raw_pca_scatter.png",
        plots_dir / "raw_pca_3d_scatter.png",
        plots_dir / "raw_pca_explained_variance.png",
        "Raw feature",
    )
    proxy_pca_summary = plot_pca(
        proxy_features,
        labels,
        plots_dir / "proxy_pca_scatter.png",
        plots_dir / "proxy_pca_3d_scatter.png",
        plots_dir / "proxy_pca_explained_variance.png",
        "Proxy feature",
    )

    # 4. Create a leakage-safe split:
    # first hold out test, then carve validation from the remaining training data.
    combined_features = feature_sets["raw_plus_proxy"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        combined_features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN_VAL,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    # Save split sizes and class counts for transparency in the final report.
    split_info = {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_class_counts": pd.Series(y_train).value_counts().sort_index().to_dict(),
        "validation_class_counts": pd.Series(y_val).value_counts().sort_index().to_dict(),
        "test_class_counts": pd.Series(y_test).value_counts().sort_index().to_dict(),
    }
    save_json(split_info, tables_dir / "train_test_split_summary.json")

    # 5. Compare the three feature sets using the smaller screening grid.
    screening_summary_rows = []

    for feature_set_name in FEATURE_SET_ORDER:
        feature_frame = feature_sets[feature_set_name]
        feature_train = feature_frame.loc[X_train.index]
        feature_val = feature_frame.loc[X_val.index]

        result_df = score_parameter_grid(feature_train, y_train, feature_val, y_val, SCREENING_GRID)
        best_row = result_df.iloc[0]
        screening_summary_rows.append(
            {
                "feature_set": feature_set_name,
                "best_val_balanced_accuracy": float(best_row["val_balanced_accuracy"]),
                "best_val_f1_macro": float(best_row["val_f1_macro"]),
                "best_params": json.dumps(
                    {
                        "n_estimators": int(best_row["n_estimators"]),
                        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
                        "max_features": best_row["max_features"],
                        "min_samples_leaf": int(best_row["min_samples_leaf"]),
                        "min_samples_split": int(best_row["min_samples_split"]),
                        "class_weight": best_row["class_weight"],
                    },
                    sort_keys=True,
                ),
            }
        )

    screening_summary_df = pd.DataFrame(screening_summary_rows).sort_values(
        by=["best_val_balanced_accuracy", "best_val_f1_macro"],
        ascending=False,
    ).reset_index(drop=True)

    save_dataframe(screening_summary_df, tables_dir / "feature_set_screening_summary.csv")
    plot_feature_set_benchmark(screening_summary_df, plots_dir / "feature_set_validation_balanced_accuracy.png")

    # 6. Tune the final Random Forest using the raw_plus_proxy representation.
    final_results_df = score_parameter_grid(X_train, y_train, X_val, y_val, FINAL_PARAM_GRID)
    save_dataframe(clean_depth_column(final_results_df), tables_dir / "final_rf_validation_results.csv")
    plot_top_validation_results(final_results_df, plots_dir / "final_rf_validation_top_configs.png")

    # Select the single best validation row.
    best_row = final_results_df.iloc[0]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "max_features": best_row["max_features"],
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "class_weight": best_row["class_weight"],
    }

    # 7. Fit two models:
    # - final_model on train+validation for the final test evaluation
    # - validation_model on train only so we can report validation performance cleanly
    final_model = make_model_pipeline(best_params)
    final_model.fit(X_train_val, y_train_val)

    validation_model = make_model_pipeline(best_params)
    validation_model.fit(X_train, y_train)

    validation_metrics = evaluate_model(validation_model, X_val, y_val)
    train_metrics = evaluate_model(final_model, X_train_val, y_train_val)
    test_predictions = final_model.predict(X_test)
    test_probabilities = final_model.predict_proba(X_test)
    test_metrics = evaluate_predictions(y_test, test_predictions)

    # Pull out the trained forest to report OOB score and tree-structure statistics.
    forest = final_model.named_steps["model"]
    oob_score = float(forest.oob_score_)
    forest_structure = summarize_forest_structure(forest)

    # 8. Save the main performance tables.
    metrics_df = pd.DataFrame(
        [
            {"split": "validation", **validation_metrics},
            {"split": "train_val", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    save_dataframe(metrics_df, tables_dir / "final_metrics.csv")

    # Save a per-class classification report for the held-out test set.
    report_df = (
        pd.DataFrame(classification_report(y_test, test_predictions, labels=class_names, output_dict=True, zero_division=0))
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )
    save_dataframe(report_df, tables_dir / "classification_report.csv")

    # Save every held-out prediction so errors can be inspected later if needed.
    prediction_df = pd.DataFrame(
        {
            "row_index": X_test.index,
            "true_label": y_test,
            "predicted_label": test_predictions,
            "max_probability": test_probabilities.max(axis=1),
        }
    ).sort_values(by=["true_label", "predicted_label", "row_index"])
    save_dataframe(prediction_df, tables_dir / "test_predictions.csv")

    # 9. Save both count-based and normalized confusion matrices.
    confusion_counts_df = compute_confusion_df(y_test, test_predictions, class_names, normalize=False)
    confusion_normalized_df = compute_confusion_df(y_test, test_predictions, class_names, normalize=True)
    save_dataframe(confusion_counts_df.reset_index().rename(columns={"index": "label"}), tables_dir / "confusion_matrix_counts.csv")
    save_dataframe(confusion_normalized_df.reset_index().rename(columns={"index": "label"}), tables_dir / "confusion_matrix_normalized.csv")
    plot_confusion(confusion_counts_df, "Held-out test confusion matrix (counts)", plots_dir / "test_confusion_matrix_counts.png", fmt="d")
    plot_confusion(confusion_normalized_df, "Held-out test confusion matrix (row-normalized)", plots_dir / "test_confusion_matrix_normalized.png", fmt=".2f")

    # 10. Save feature-importance outputs so the final model is at least somewhat interpretable.
    feature_importance_df = build_feature_manifest(raw_features, proxy_features)
    feature_importance_df["importance"] = forest.feature_importances_
    feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    save_dataframe(feature_importance_df, tables_dir / "feature_importance.csv")
    plot_feature_importance(feature_importance_df, plots_dir / "top_feature_importance.png")

    # 11. Save the trained pipeline so preprocessing and model can be reloaded together.
    with (model_dir / "final_random_forest_pipeline.pkl").open("wb") as file:
        pickle.dump(final_model, file)

    # Save one compact JSON file with the most important settings and results.
    model_info = {
        "data_path": DATA_PATH.resolve(),
        "random_state": RANDOM_STATE,
        "primary_metric": PRIMARY_METRIC,
        "feature_representation": "raw_plus_proxy",
        "best_params": best_params,
        "best_cv_row": {
            **best_row.to_dict(),
            "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        },
        "forest_structure": forest_structure,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "oob_score": oob_score,
        "raw_pca_summary": raw_pca_summary,
        "proxy_pca_summary": proxy_pca_summary,
    }
    save_json(model_info, model_dir / "model_info.json")

    # Save a short markdown summary that is easy to read at a glance.
    run_summary = build_run_summary(
        DATA_PATH.resolve(),
        class_counts,
        screening_summary_df,
        best_params,
        forest_structure,
        train_metrics,
        test_metrics,
        oob_score,
        feature_importance_df,
    )
    save_text(run_summary, output_dir / "run_summary.md")

    # Print the headline results in the terminal too.
    print("Capstone final Random Forest run complete")
    print("---------------------------------------")
    print(f"Dataset: {DATA_PATH.resolve()}")
    print(f"Output directory: {output_dir}")
    print(f"Best params: {best_params}")
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={test_metrics['f1']:.4f}"
    )
    print(f"OOB score: {oob_score:.4f}")


if __name__ == "__main__":
    main()

```