# Capstone Postmortem

## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection Postmortem
* **Data Modality:** Tabular
* **Task Type:** Primarily classification, with one unsupervised structure-discovery phase
* **One-Sentence Goal:** Use measured phase currents and voltages from a simulated three-phase electrical system to detect whether a fault is present and, when possible, classify the exact fault pattern.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Capstone retrospective across multiple machine learning approaches: preprocessing and leakage-safe evaluation, logistic regression, threshold tuning, regularization, Naive Bayes, k-Nearest Neighbors, SVMs, decision trees, ensembles, PCA + k-means, and multilayer perceptrons.
* **Key Assumptions of This Technique:**
  * The six measured signals `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc` contain enough information to separate normal operation from faulted operation, and possibly to separate different fault patterns.
  * Rows are independent enough that random stratified train/validation/test splits are meaningful.
  * The observed label structure can be studied in two ways: a simpler binary `any fault` task and a harder multiclass fault-pattern task.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

This capstone ended up being a strong example of why it is useful to compare multiple model families instead of committing too early to one idea. The electrical dataset is small enough to let me try many methods, but structured enough that those methods behaved differently for meaningful reasons. Some methods told me more about optimization, some told me more about geometry, and some told me more about the target framing itself. In that sense, the full semester of techniques was a better fit than any single model.

The most important assumption that changed over time was not about optimization, but about the task definition. Early in the project I often used the binary `any fault` target because it was simple and let me isolate training behavior. Later, I realized that this target was often too easy, and that the multiclass fault-pattern problem revealed much more about the real strengths and weaknesses of each method. That shift was one of the most important lessons of the capstone.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** The core representation throughout the project was the six measured electrical features `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc`.

* **Why this representation was reasonable for this week:**

These six values are the closest thing in the dataset to real sensor inputs, so they were the most honest representation to build around. I intentionally did **not** use the fault-indicator columns `G`, `C`, `B`, and `A` as model inputs, because those columns define the label and would leak the answer directly into the model.

I used two main target representations:

* **Binary proxy target:** `fault` vs `no fault`, where any active indicator counts as a fault
* **Multiclass target:** one label formed by concatenating the four fault bits into patterns such as `0000`, `0110`, `0111`, `1001`, `1011`, and `1111`

This ended up being a very useful design choice. The binary target was helpful for early baselines, threshold studies, and testing whether the measurements contain obvious signal at all. The multiclass target was much closer to the actual project objective, because it asked the model to distinguish among several meaningful electrical states rather than only answering whether anything abnormal happened.

The unsupervised phase also used the same six measured features, but without labels during clustering. That was important because it let me compare the geometry of the measurement space to the known fault categories after the fact instead of accidentally building the labels into the clustering step.

---

## 4. What Was Attempted

This capstone developed in stages rather than through one single model. The overall system design was:

* Load `electrical_fault_data.csv`
* Split into train / validation / test with reproducible seeds
* Fit preprocessing on train only
* Use the six measurements as features
* Train either a binary classifier, a multiclass classifier, or an unsupervised model
* Evaluate on validation first, then lock choices before touching the test split
* Interpret the results in terms of both model behavior and the structure of the electrical problem

The main methods I explored were:

* **Preprocessing and leakage control:** early work focused on correct splitting, scaling, and keeping the fault-bit columns out of the feature matrix.
* **Logistic regression baseline:** this gave me a simple linear model to understand loss behavior and establish a baseline. It was useful mainly as a reference point.
* **Threshold tuning for binary classification:** this showed that model quality is not just about the raw classifier, but also about how the operating threshold changes false positives and false negatives.
* **Regularization sweeps:** these helped test whether poor performance came from overfitting or from representational limits. In this project, regularization usually did not change much.
* **Gaussian Naive Bayes vs kNN:** this comparison showed that neighborhood structure in the data mattered much more than conditional-independence assumptions.
* **Linear SVM vs RBF SVM:** this was one of the clearest demonstrations that nonlinear boundaries fit the binary fault task much better than linear ones.
* **Decision tree, Random Forest, and Gradient Boosting:** these explored nonlinear, rule-based representations and ensemble effects on the harder multiclass task.
* **PCA and k-means clustering:** this was used to study the geometry of the measurement space and compare unsupervised structure to known fault labels.
* **MLPs / neural networks:** these tested whether learned nonlinear feature combinations could outperform simpler methods, especially on multiclass prediction.
* **Final multiclass comparison:** I directly compared kNN, RBF SVM, Random Forest, and MLP on the same multiclass split so the final judgment would come from a fair side-by-side setup.

Training setup and constraints stayed fairly consistent:

* Dataset size was `7,861` rows
* Feature count was only `6`, so this was a compact tabular problem rather than a high-dimensional one
* Known fault patterns observed in the data were `0000`, `0110`, `0111`, `1001`, `1011`, and `1111`
* I kept preprocessing leakage-safe by fitting imputation and scaling on training data only
* I used validation performance for model selection and saved test evaluation for the end
* The data is simulated, which means strong performance should be interpreted carefully because the real-world noise and drift of an actual grid system may be more complex

What I intentionally did not attempt:

* I did not treat the project as a time-series problem because the dataset is row-based tabular data rather than a sequential waveform dataset.
* I did not use the fault-indicator bits as inputs.
* I did not do extremely large hyperparameter searches, because the capstone goal was to learn from method behavior rather than to chase leaderboard-style optimization.

---

## 5. Results or Observations

Several results stood out across the semester.

### Dataset and structure observations

* The dataset had `7,861` rows and six measurement features.
* The fault-pattern distribution was fairly balanced across the six observed classes, though not perfectly uniform.
* PCA showed strong low-dimensional structure:
  * `PC1 = 30.74%`
  * `PC2 = 25.70%`
  * first two PCs combined = `56.44%`
  * first four PCs combined = `97.21%`

This was important because it showed that the electrical measurements are highly structured rather than noisy or random.

### Binary-task observations

The binary `any fault` task was useful, but it often turned out to be easier than the full project goal.

* **Logistic regression baseline:** around `0.6992` accuracy, `1.0000` recall, and `0.8229` F1 in one early setup, which showed that a linear model could detect many faults but behaved in a very one-sided way.
* **Threshold-tuned logistic regression:** one thresholding policy produced only `0.5340` accuracy and `0.5391` balanced accuracy on test, which reinforced how fragile the binary operating point could be when the model itself was weak.
* **Regularization study:** tuning `C` and related settings barely changed performance, which suggested that model capacity control was not the main bottleneck for the logistic setup.
* **GaussianNB:** strong but not dominant, with validation F1 around `0.9810`.
* **kNN on the binary task:** essentially perfect, with validation scores at or near `1.0000` and final test accuracy/F1 of `1.0000` for the selected configuration.
* **RBF SVM on the binary task:** also nearly perfect, with final test accuracy `0.9968`, balanced accuracy `0.9977`, and F1 `0.9977`.

The pattern here was consistent: once I allowed nonlinear or local-decision methods, binary fault detection became very easy.

### Multiclass-task observations

The multiclass fault-pattern task was more informative because it exposed real differences between methods.

* **Decision Tree:** final test accuracy `0.8798`, balanced accuracy `0.8590`, macro F1 `0.8585`
* **Neural network (one multiclass MLP run):** test accuracy `0.8525`, macro F1 `0.8225`
* **Later deep-learning assignment:** validation accuracy `0.8592`, test accuracy `0.8567`, macro F1 `0.7855`
* **Final multiclass comparison**
  * **kNN:** accuracy `0.8887`, balanced accuracy `0.8691`, macro F1 `0.8691`
  * **Random Forest:** accuracy `0.8792`, balanced accuracy `0.8579`, macro F1 `0.8575`
  * **RBF SVM:** accuracy `0.8538`, balanced accuracy `0.8275`, macro F1 `0.8254`
  * **MLP:** accuracy `0.8449`, balanced accuracy `0.8172`, macro F1 `0.8143`

The final comparison selected **kNN** as the best model by validation balanced accuracy, and it also produced the strongest held-out test performance among the four compared families.

### Unsupervised observations

The unsupervised work was valuable, but in a different way from the supervised models.

* Best tested k-means setting by silhouette score was `k=8`
* Silhouette score at `k=8` was `0.4006`
* Post hoc adjusted Rand index at `k=8` was only `0.1688`

That means the measurement space clearly has structure, but the natural geometric groupings found by k-means only weakly aligned with the human-defined fault labels.

### Recurring qualitative behavior

* The binary task was often almost too easy once I moved beyond linear methods.
* The multiclass problem consistently revealed confusion between `0111` and `1111`, even when overall performance was good.
* More complex models did **not** automatically perform better. In particular, Random Forest did not clearly beat the best single decision tree in one assignment, and the final MLP did not beat kNN.
* Perfect or near-perfect train performance for kNN and Random Forest in the final comparison showed that overfitting risk was real, but these methods still generalized reasonably well on test because the data itself appears highly structured.

---

## 6. Interpretation and Judgment

The biggest thing I learned from this capstone is that machine learning success depends as much on **problem framing** as on model choice. Early in the semester, I kept asking whether a particular method was good or bad. By the end, I think the more important question was: "good or bad for which target?" On the binary `any fault` task, many nonlinear methods looked almost perfect. On the multiclass fault-pattern task, the differences between methods became much more meaningful. That taught me that easy proxy tasks can be useful for early experimentation, but they can also hide the real difficulty of the project.

I also learned that strong performance does not always mean a model has discovered a deep or general truth. Some of the binary results were so good that they forced me to question leakage, check the pipeline carefully, and think about whether the task itself was too simple. That was a healthy lesson. A result can be numerically excellent and still require skepticism. In this project, the strongest explanation was usually not leakage, but that the electrical measurements really do separate fault vs no-fault very clearly in the simulated data.

Another important lesson was that **interpretability and accuracy trade off in different ways depending on the method**. Logistic regression and linear SVMs were easier to reason about, but they often underfit the structure of the data. RBF SVMs, trees, kNN, and MLPs captured more nonlinear behavior, but they were harder to interpret directly. The unsupervised work reinforced this by showing that the data has real structure, but that structure is not the same thing as the class labels. In other words, geometry, prediction, and explanation are related, but they are not identical goals.

I also came away with a better understanding of evaluation. Metrics are not interchangeable. Threshold selection changed the binary confusion matrix dramatically. Balanced accuracy mattered more once I cared about multiclass fairness across fault patterns. Macro metrics were more honest than plain accuracy in the multiclass setting. Looking only at one metric would have hidden important details.

Overall, I think my final approach was a good fit for the dataset, especially once I reframed the project around multiclass fault-pattern classification instead of staying only with the binary proxy. The final comparison suggests that the data is well suited to models that exploit local neighborhood structure, since kNN ended up strongest. At the same time, the recurring confusion between `0111` and `1111` shows that even when the data is highly structured, some electrical states still overlap in a way that limits clean separation from only six measured features.

---

## 7. Forward-Looking Adjustment

If I continued this project, I would keep the same leakage-safe evaluation discipline, but I would change the next phase in a few specific ways:

* Keep the multiclass fault-pattern framing as the main task, because it is much more informative than the binary proxy.
* Add physically motivated engineered features such as phase differences, current/voltage contrasts, or ratio-style summaries.
* Focus error analysis on the repeated `0111` vs `1111` confusion to understand whether those classes are physically similar or just poorly represented by the existing six features.
* Test whether richer representations help the MLP more than they help kNN, since the current raw-feature setting seems to favor neighborhood methods.
* If more realistic data were available, validate everything again under noisier or shifted conditions before trusting the strongest current results.

If data or resources were not constrained, the next major step would be to move from static row classification toward a more realistic stream or time-window formulation, since real electrical fault detection is likely to involve temporal behavior rather than isolated snapshots.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

Several methods were only partial fits, and those mismatches were actually some of the most educational parts of the project.

* **Logistic regression** was a useful baseline, but it was not expressive enough for the real structure of the task.
* **Threshold tuning** showed that operational choices matter, but it also showed that no threshold can rescue a weak underlying model.
* **Naive Bayes** required adaptation from the course example because count-based assumptions did not match continuous electrical features.
* **k-means clustering** found real geometric structure, but that structure did not align closely with the supervised labels, as shown by the low adjusted Rand index.
* **Ensembles and neural nets** were not automatically superior just because they were more sophisticated.

The value of these mismatches is that they taught me a broader machine-learning lesson: when a method underperforms, the issue is not always "bad model." Sometimes the issue is the target, the assumptions, the representation, or the meaning of the evaluation metric. This dataset was especially good for learning that lesson because different methods failed for different reasons, and those reasons were interpretable.
