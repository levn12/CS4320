CS 4320 Levi Dockstader -- 1/26/2026

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

Not applicable at this stage. 

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

Since this week's work was framing my problem, I decided to use this section of the template for my observations about the data I'm working with. With my electrical fault tabular dataset, I will be developing a classification model that receives voltage and current values for the three phases of a power system as input, and predict whether a the system has encountered an electrical fault and what type of fault has occured. I'm trying to develop a tool that can reliably detect and classify electrical faults in a power grid when they happen.

The real world application of this problem seems pretty straightforward: there are faults that occur in electrical grids, and power companies must be able to reliably detect when and where those faults occur based off of readings from sensors in equipment. Even though the data I'm working with is technically simulated and may not be as affected by noise and disturbances as a real-world system, I think it mimics the behavior closely enough to train a model that can at least be a proof-of-concept for the type of solution a power company might pursue in trying to find a more reliable fault detection system. Aside from real-time fault detection, a model like the one I'm framing could easily be expanded to find other trends if more real-world features were associated with each data point. For example, maybe a specific type of equipment or location tends to fail more often. If a model is able to classify faults, it could then also group those faults based on equipment type, location, or some other insightful feature. The model could assist a company in choosing a new type of equipment that fails less frequently or identify common issues between faults over time.

This model will be supervised in its training because the dataset I'm working with is tabular and already labeled. This could make training easy, but I think the simplicity of the values tracked also limits how useful the model may be, because there's not much room for predicting faults ahead of time, seeing what parts of equipment may be failing, or other insights about the causes and effects of electrical faults; it simply would just detect and classify an anomaly data point. That being said, I don't think there is much uncertainty associated with this framing of the data: There is a strong cause-effect relation between voltage/current and electrical fault behavior; all the data is already correctly labeled; the target result is not a proxy and it's well-defined. The insights available may be limited, but at least the framing is clear.

I think this perspective is probably the most obvious framing. With the data available in this set, I don't think it would be possible to find a way to predict faults beforehand -- only detect them after-the-fact. However, one could make a regression model instead of a classification model. I plan on giving my model voltage/current values and have it tell me if there has been a fault. You could also build a model to tell it there has been a certain type of fault and have it predict what the voltage/current values likely were associated with that fault. However, I don't think that's the best option for these particular data because a fairly wide range of voltage/current values may exist for any one type of fault, and there's not more features to go off of in predicting where the values may lie in the possible range. Additionally, in the real world it's more likely that you're trying to detect the fault based on measured values rather than predicting what values you should measure after detecting a fault through some other means.

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
