# ✈️ Jet Engine Hospital
### Leakage-Safe Multi-Task Predictive Maintenance for NASA C-MAPSS Turbofan Engines

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Jet Engine Hospital** is an end-to-end predictive maintenance system built on the **NASA C-MAPSS turbofan engine degradation dataset**.

The project combines:

- Remaining Useful Life (**RUL**) prediction
- Near-term failure classification
- Unsupervised anomaly detection
- Time-series feature engineering
- Probability calibration
- Conformal uncertainty estimation
- Early-warning analysis
- Maintenance-oriented decision support
- Interactive deployment with **Streamlit**

The emphasis is not only on predictive performance, but also on **data leakage prevention, reproducibility, uncertainty, calibration, and operationally meaningful evaluation**.

---

## 📌 Project Overview

Predictive maintenance aims to detect degradation before equipment reaches a critical failure state.

For turbofan engines, an effective monitoring system should answer several related questions:

1. **How many operational cycles remain before failure?**
2. **Is the engine likely to fail within the next 10, 20, or 30 cycles?**
3. **Is the current engine behavior abnormal compared with healthy operation?**
4. **How confident is the system in its predictions?**
5. **Should maintenance action be considered?**

Jet Engine Hospital addresses these questions through a unified machine learning workflow.

---

## 🎯 Project Objectives

The project was designed to:

- Estimate **Remaining Useful Life (RUL)**
- Predict failure within **10-, 20-, and 30-cycle horizons**
- Detect abnormal engine behavior without using failure labels during anomaly-detector fitting
- Build causal time-series features
- Prevent engine-level and temporal data leakage
- Compare linear and nonlinear machine learning models
- Optimize classification thresholds using validation data
- Calibrate probabilistic predictions
- Estimate uncertainty using conformal prediction
- Measure warning lead time before failure
- Compare anomaly detection approaches
- Export trained artifacts for deployment
- Provide interpretable maintenance recommendations

---

# 📊 Dataset

## NASA C-MAPSS

The project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** turbofan engine degradation benchmark.

Each engine is represented as an independent multivariate time series.

Each engine-cycle observation contains:

- `engine_id`
- cycle number
- 3 operational settings
- 21 sensor measurements

Training trajectories continue until failure, while test trajectories stop before failure.

The project works with C-MAPSS subsets including:

- **FD001** — one operating condition, one fault mode
- **FD003** — one operating condition, multiple fault modes
- **FD004** — multiple operating conditions and multiple fault modes

The Streamlit dashboard provides exploration for FD001, FD003, and FD004.

---

# 🧠 Machine Learning Tasks

The system is built around three complementary predictive tasks.

---

## 1️⃣ Remaining Useful Life Regression

The regression task estimates the number of cycles remaining before an engine reaches failure.

For an engine observed at cycle `t`, RUL represents the remaining operational lifetime.

### Models evaluated

- Ridge Regression
- Polynomial Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Both current-cycle and historical window-based representations were investigated.

### Evaluation Metrics

- Mean Absolute Error (**MAE**)
- Root Mean Squared Error (**RMSE**)
- R²
- Maintenance-oriented asymmetric error score
- Near-failure MAE
- Per-engine prediction traces

### Best Validation Configuration

A strong configuration used:

```text
Model: Ridge Regression
Representation: Window-based features
RUL Cap: 100
Alpha: 0.1
Features Used: 121
```

Approximate validation performance:

| Metric | Result |
|---|---:|
| MAE | 11.20 cycles |
| RMSE | 13.92 cycles |
| R² | 0.77 |
| Near-failure MAE | 9.40 cycles |

The results demonstrate that a relatively simple regularized linear model can perform competitively when combined with carefully constructed time-series features.

---

## 2️⃣ Failure Horizon Classification

The second task estimates the probability that an engine will fail within different future horizons.

Three binary targets are constructed:

```text
Failure within 10 cycles
Failure within 20 cycles
Failure within 30 cycles
```

This produces three separate risk estimates rather than one reused binary label.

### Models evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting

Logistic Regression is used as a transparent baseline and compared with nonlinear alternatives.

### Evaluation Metrics

- Precision
- Recall
- F1-score
- PR-AUC
- Confusion Matrix
- Brier Score
- Calibration Curves

Because positive failure cases can be imbalanced, **PR-AUC** is treated as an important evaluation metric.

### Test Performance

Approximate PR-AUC results:

| Failure Horizon | PR-AUC |
|---|---:|
| 10 cycles | ~0.88 |
| 20 cycles | ~0.73 |
| 30 cycles | ~0.81 |

Recall was above approximately **0.92** across the evaluated horizons after validation-based threshold selection.

### Selected Decision Thresholds

The classification threshold was not automatically fixed at `0.50`.

Approximate validation-selected thresholds were:

| Horizon | Threshold |
|---|---:|
| 10 cycles | 0.113 |
| 20 cycles | 0.223 |
| 30 cycles | 0.117 |

Thresholds were chosen using the validation data and the declared asymmetric maintenance cost policy.

---

## 3️⃣ Unsupervised Anomaly Detection

A separate pipeline investigates whether degradation can be identified **without using failure labels during anomaly-detector fitting**.

The detectors are trained using a healthy proxy region derived only from training engines.

### Methods evaluated

- Isolation Forest
- Local Outlier Factor (**LOF**)
- One-Class SVM
- PCA Reconstruction Error

Because different anomaly detectors produce scores on different scales, the raw values are converted into comparable ranks or percentiles.

The normalized convention is:

```text
Higher anomaly score = more abnormal engine behavior
```

The project then examines whether anomaly evidence increases as engines approach failure.

---

# ⏳ Time-Series Feature Engineering

Features are constructed causally.

For a prediction made at cycle `t`, only information available at or before cycle `t` is allowed.

Candidate features include:

- Current sensor measurements
- Rolling mean
- Rolling standard deviation
- Rolling minimum and maximum
- Rolling slopes
- First differences
- Exponentially weighted statistics
- Sensor ratios
- Operating-condition-aware features
- Window-based historical representations

A central rule of the project is:

> Future engine cycles must never affect features calculated for an earlier cycle.

Centered rolling windows, final-cycle-derived features, or whole-engine transformations that use future information would introduce leakage and are therefore avoided.

---

# 🛡️ Leakage-Safe Experimental Design

A major design requirement is preventing **data leakage**.

The train, validation, and test split is performed at the **engine level**.

```text
Engine 1 ───────────────────► Train
Engine 2 ───────────────────► Train
Engine 3 ───────────────────► Validation
Engine 4 ───────────────────► Test
```

All observations belonging to a single engine remain in exactly one split.

This prevents windows from the same engine trajectory from appearing in both training and evaluation sets.

---

## Preprocessing Rule

All learned preprocessing operations are fitted using training engines only.

```text
Engine-Level Split
        ↓
Fit Preprocessing on Training Engines
        ↓
Causal Feature Engineering
        ↓
Model Training
        ↓
Validation / Tuning
        ↓
Calibration
        ↓
Locked Final Evaluation
```

This applies to learned transformations such as:

- scalers,
- feature selectors,
- PCA models,
- condition-based transformations,
- anomaly detectors,
- probability calibrators.

---

# 🧪 Validation and Calibration

Model selection and probability calibration are separated from the final evaluation.

The project uses validation data for tasks such as:

- hyperparameter selection,
- RUL-cap selection,
- classification threshold selection,
- probability calibration,
- anomaly threshold selection,
- uncertainty calibration.

The final test data is reserved for the locked final evaluation.

---

# ⚖️ Asymmetric Maintenance Cost

Not every RUL prediction error has the same operational consequence.

For predictive maintenance:

```text
Predicting failure too early
        ↓
Possible unnecessary inspection or maintenance

Predicting failure too late
        ↓
Potentially dangerous delayed intervention
```

Therefore, late predictions are penalized more strongly than equally sized early predictions.

For a prediction error:

```text
error = prediction - true RUL
```

a positive error represents RUL overestimation and therefore a **late warning**.

---

# 🚨 Early-Warning Analysis

Traditional row-level classification metrics do not fully describe the usefulness of an early-warning system.

The project also evaluates warning behavior at the engine level.

Important quantities include:

- First persistent warning cycle
- Lead time before failure
- Miss rate
- False-alert behavior
- Early-warning burden
- Late-warning delay
- Warning persistence

Conceptually:

```text
First Persistent Alert
        ↓
──────────── Lead Time ────────────
        ↓                         ↓
     Warning                   Failure
```

The objective is not simply to classify individual rows correctly, but to provide useful warning time before failure.

---

# 🎯 Probability Calibration

Raw classifier scores are not necessarily reliable probabilities.

Probability calibration is used to improve agreement between:

```text
Predicted failure probability
              ↓
Observed failure frequency
```

Calibration is evaluated using:

- Calibration curves
- Brier Score
- Validation-based calibration procedures

Reliable probabilities are particularly important because the final maintenance decision uses probability thresholds.

---

# 📏 Conformal Prediction

The RUL pipeline also incorporates uncertainty estimation using **conformal prediction**.

Rather than returning only a point prediction such as:

```text
Predicted RUL = 35 cycles
```

the system can provide an interval around the estimate.

Conceptually:

```text
Lower Bound ≤ Predicted RUL ≤ Upper Bound
```

Residuals from a dedicated calibration set are used to construct prediction intervals without requiring a specific residual distribution assumption.

The deployed dashboard provides a **90% RUL interval**.

---

# 🔄 Engine-Level Bootstrap Confidence Intervals

Uncertainty in evaluation metrics is estimated using bootstrap resampling.

Because rows from the same engine are statistically dependent, the project uses:

> **Engine-level bootstrap resampling**

instead of independently resampling rows.

This preserves complete engine trajectories and provides more appropriate confidence intervals for engine-level evaluation.

The evaluation uses **1,000 bootstrap samples** where applicable.

---

# 🔍 Latent Degradation Structure

PCA and clustering are also used to investigate hidden degradation structure.

Two latent groups were identified during exploratory analysis.

They are referred to as:

```text
latent_A
latent_B
```

rather than being interpreted as known physical fault modes.

This distinction is important because the dataset does not provide ground-truth labels proving that the clusters correspond directly to specific physical faults.

---

# 🏥 Maintenance Decision Support

The final decision layer combines several evidence sources:

```text
RUL Prediction
       +
Failure Probability
       +
Anomaly Evidence
       +
Prediction Uncertainty
```

to produce an interpretable maintenance state.

### Possible Recommendations

| Status | Meaning |
|---|---|
| 🟢 **CONTINUE** | Current evidence does not indicate immediate intervention |
| 🟠 **INSPECT** | Elevated risk, anomaly, or uncertainty suggests inspection |
| 🔴 **STOP** | A validated critical condition has been triggered |

The recommendation logic remains traceable to underlying model outputs.

---

# 🖥️ Streamlit Dashboard

The repository includes an interactive Streamlit application.

The deployment-ready Community Cloud version is located in:

```text
app2/
├── artifacts/
├── .gitignore
├── README.md
├── app.py
└── requirements.txt
```

The application uses **bundled, precomputed NASA C-MAPSS inference artifacts**.

Therefore, the deployed dashboard does not require:

- retraining the models,
- external services,
- API keys,
- application secrets.

---

## Dashboard Functionality

The Streamlit application supports:

- FD001, FD003, and FD004 dataset selection
- Engine selection
- Cycle-level exploration
- Sensor exploration
- Sensor timelines
- Remaining Useful Life timelines
- **90% RUL prediction interval**
- Calibrated 10-cycle failure risk
- Calibrated 20-cycle failure risk
- Calibrated 30-cycle failure risk
- Normalized anomaly evidence
- Evidence-family agreement
- Maintenance recommendations:
  - CONTINUE
  - INSPECT
  - STOP

---

# 📁 Repository Structure

```text
ml-final-project/
│
├── app/
│   ├── artifacts/
│   ├── README.md
│   ├── app.py
│   └── requirements.txt
│
├── app2/
│   ├── artifacts/
│   ├── .gitignore
│   ├── README.md
│   ├── app.py
│   └── requirements.txt
│
├── artifacts/
│   └── Saved models, preprocessing objects,
│       calibrators, thresholds, and metadata
│
├── data/
│   └── Project data and prepared inputs
│
├── figures/
│   └── Generated evaluation plots
│
├── notebooks/
│   └── Analysis and experimental notebooks
│
├── report/
│   └── Technical project report
│
├── src/
│   └── Reusable project source code
│
└── requirements.txt
```

`app2/` contains the **Streamlit Community Cloud deployment version** of Jet Engine Hospital.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/MohaddesehGanjkhanlou/ml-final-project.git
cd ml-final-project
```

---

## Create a Virtual Environment

### Windows

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

---

# ▶️ Run the Streamlit Dashboard Locally

The deployment application uses **Python 3.12**.

Install its dependencies:

```bash
python -m pip install -r app2/requirements.txt
```

Start the application:

```bash
streamlit run app2/app.py
```

Streamlit will display the local application URL in the terminal.

---

# ☁️ Streamlit Community Cloud Deployment

The application is prepared for deployment through **Streamlit Community Cloud**.

Use the following configuration:

```text
Repository:
MohaddesehGanjkhanlou/ml-final-project

Branch:
main

Main file path:
app2/app.py

Python:
3.12
```

No secrets are required.

The deployment-specific dependency file is located beside the application entry point:

```text
app2/requirements.txt
```

---

# 🧩 Deployment Architecture

```text
NASA C-MAPSS Data
        ↓
Leakage-Safe ML Pipeline
        ↓
Validated Models
        ↓
Exported Inference Artifacts
        ↓
app2/artifacts/
        ↓
Streamlit Dashboard
        ↓
Engine + Cycle Selection
        ↓
┌────────────────────────────────┐
│ RUL Prediction                 │
│ 90% Prediction Interval        │
│ 10 / 20 / 30 Cycle Risk        │
│ Anomaly Evidence               │
│ Evidence-Family Agreement      │
└────────────────────────────────┘
        ↓
CONTINUE / INSPECT / STOP
```

The deployment layer is intentionally separated from training so that the dashboard performs inference using previously exported artifacts.

---

# 🔬 Reproducibility

The project was designed to support reproducible experimentation.

Important practices include:

- Fixed random seeds
- Engine-level splitting
- Causal time-series feature generation
- Training-only preprocessing
- Explicit train / validation / test separation
- Validation-only hyperparameter tuning
- Validation-based threshold selection
- Dedicated calibration procedures
- Saved inference artifacts
- Artifact reload checks
- Prediction parity checks
- Engine-level bootstrap confidence intervals

---

# 📐 Evaluation Philosophy

The system intentionally avoids choosing models using only one metric.

Different components require different evaluation criteria.

### RUL Regression

```text
MAE
RMSE
R²
Asymmetric PHM Cost
Near-Failure Error
Engine-Level Traces
Prediction Interval Coverage
```

### Failure Classification

```text
PR-AUC
Precision
Recall
F1
Brier Score
Confusion Matrix
Calibration Curves
```

### Anomaly Detection

```text
Anomaly Score Trajectories
Lead Time
False Alerts
Miss Rate
Threshold Stability
Detector Agreement
```

### Maintenance Decision Layer

```text
Warning Lead Time
Early-Warning Burden
Late-Warning Delay
Decision Consistency
Maintenance-Oriented Cost
```

---

# ⚠️ Limitations

This project uses a **simulated turbofan degradation benchmark**.

Therefore:

- The results should not be interpreted as direct validation on real aircraft engines.
- Simulated sensor behavior cannot represent every real operational condition.
- Maintenance thresholds depend on the assumed decision-cost policy.
- Anomaly scores indicate unusual behavior but do not automatically identify a physical fault.
- Latent clusters should not be interpreted as known fault modes without ground-truth labels.
- Predictive performance on C-MAPSS does not guarantee performance in production aviation environments.

This repository is an educational and machine-learning research project, not a production aviation safety system.

---

# 🚀 Possible Future Improvements

Potential extensions include:

- LSTM-based sequence models
- Temporal Convolutional Networks
- Transformer-based time-series models
- Survival analysis
- Domain adaptation across C-MAPSS subsets
- Improved operating-condition normalization
- Sensor-level explainability
- Model monitoring
- Dockerized deployment
- REST API inference
- Automated testing
- Continuous integration
- Experiment tracking

---

# 🛠️ Technologies

## Programming

- Python
- Jupyter Notebook

## Data Processing

- NumPy
- Pandas

## Machine Learning

- Scikit-learn
- Ridge Regression
- Polynomial Regression
- Random Forest
- Gradient Boosting
- Logistic Regression
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- PCA

## Machine Learning Concepts

- Regression
- Classification
- Unsupervised Learning
- Time-Series Feature Engineering
- Predictive Maintenance
- Probability Calibration
- Anomaly Detection
- Conformal Prediction
- Bootstrap Confidence Intervals
- Threshold Optimization

## Visualization

- Matplotlib

## Deployment

- Streamlit
- Streamlit Community Cloud

## Development Tools

- Git
- GitHub
- Jupyter
- VS Code

---

# 👩‍💻 Author

**Mohaddeseh Ganjkhanlou**

Computer Science

Interested in:

- Software Development
- Machine Learning
- Data-Driven Systems

GitHub: [@MohaddesehGanjkhanlou](https://github.com/MohaddesehGanjkhanlou)

---

# 📚 Acknowledgements

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**, a widely used benchmark for prognostics and predictive maintenance research.

The project was developed as a **Machine Learning Capstone Project** at **Shahid Beheshti University**.

The project focuses on building a defensible predictive-maintenance workflow rather than optimizing a single headline metric.

---

# ⭐ Project at a Glance

```text
Raw Turbofan Sensor Data
          ↓
Engine-Level Data Split
          ↓
Leakage-Safe Preprocessing
          ↓
Causal Time-Series Features
          ↓
 ┌────────────────────┬─────────────────────┬─────────────────────┐
 │ RUL Regression     │ Failure Prediction  │ Anomaly Detection   │
 │                    │ 10 / 20 / 30 cycles │                     │
 └────────────────────┴─────────────────────┴─────────────────────┘
          ↓
Calibration + Uncertainty
          ↓
Engine-Level Evaluation
          ↓
Early-Warning Analysis
          ↓
Maintenance Decision Support
          ↓
Streamlit Dashboard
```

**Jet Engine Hospital demonstrates an end-to-end machine learning workflow where predictive performance, uncertainty, leakage prevention, calibration, interpretability, and operational decision-making are evaluated together.**

---

## ⭐ If you find this project useful

Feel free to explore the notebooks, source code, evaluation results, and Streamlit application.
