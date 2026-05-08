# Maritime Pirate Attack Risk Analysis

A machine learning pipeline that analyses historical pirate attack incidents (1994–2020),
predicts attack occurrence probabilities, and serves results through an interactive
Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
4. [Setup](#4-setup)
5. [Usage](#5-usage)
6. [Models and Performance](#6-models-and-performance)
7. [Output Description](#7-output-description)
8. [Dashboard](#8-dashboard)
9. [Testing](#9-testing)
10. [Project Structure](#10-project-structure)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

This project transforms raw incident reports into actionable risk intelligence by:

- Engineering temporally and geographically meaningful features from sparse incident data.
- Training a **Gradient Boosting Regressor** to predict `log_shore_distance` (how far from
  shore an attack occurs), capturing attack opportunism patterns.
- Training a **Gradient Boosting Classifier** to estimate the probability that any given
  incident constitutes a confirmed attack.
- Assigning every incident to a risk band (Low / Moderate / High / Critical).
- Displaying all results through a tabbed Streamlit dashboard with Plotly visualisations.

---

## 2. Dataset

**Source file:** `pirate_attacks_clean.csv`

| Property | Value |
|---|---|
| Rows | 6,555 incidents |
| Columns | 9 |
| Time range | 1994 – 2020 |
| Geographic coverage | Global |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `year` | int | Year of incident |
| `month` | int | Month of incident (1–12) |
| `longitude` | float | Incident longitude (decimal degrees) |
| `latitude` | float | Incident latitude (decimal degrees) |
| `attack_type` | str | Nature of the attack |
| `vessel_status` | str | Vessel operational status |
| `shore_distance` | float | Distance to shore in nautical miles |
| `nearest_country` | str | ISO code of nearest country |
| `region` | str | World Bank region |

---

## 3. Architecture

```
pirate_attacks_clean.csv
        |
        v
  load_data()
        |
        v
  preprocess_data()          <- filters invalid coords, fills unknowns
        |
        v
  engineer_features()        <- 12 derived features (log, haversine, cyclical, ordinal)
        |
        +---------------------------+
        |                           |
        v                           v
  Regression pipeline         Classification pipeline
  (log_shore_distance)        (attack_occurred)
        |                           |
  run_kfold_cv()              run_classification_cv()
  fit_final_model()           generate_attack_probability_column()
        |                           |
        +---------------------------+
                    |
                    v
     pirate_attacks_with_probability.csv
     (adds: attack_occurred, attack_probability_pct, risk_band)
```

---

## 4. Setup

**Prerequisites:** Python 3.10 or higher.

```bash
# Clone or download the project
cd zain_prob

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Usage

**Run the ML pipeline:**

```bash
python app.py
```

This executes all seven pipeline steps and writes `pirate_attacks_with_probability.csv`
to the project directory.

**Launch the interactive dashboard:**

```bash
streamlit run gui.py
```

Open `http://localhost:8501` in a browser.

---

## 6. Models and Performance

Both models use identical hyperparameters evaluated via 10-fold cross-validation.

### Regression Model — GradientBoostingRegressor

Target: `log_shore_distance`

| Metric | Mean (10-fold CV) |
|---|---|
| R² (test) | 0.7524 |
| MAE (test) | 0.61 |
| RMSE (test) | 0.82 |

### Classification Model — GradientBoostingClassifier

Target: `attack_occurred` (binary)

| Metric | Mean (10-fold Stratified CV) |
|---|---|
| AUC-ROC (test) | 0.9044 |
| Accuracy (test) | 0.8468 |
| Brier Score (test) | 0.11 |

### Shared Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `min_samples_leaf` | 10 |
| `random_state` | 42 |

### Preprocessing (inside pipeline, fitted on training folds only)

| Feature type | Transformers |
|---|---|
| Numeric | SimpleImputer (median) + StandardScaler |
| Categorical | SimpleImputer (constant) + OneHotEncoder (drop=first) |

---

## 7. Output Description

The pipeline produces `pirate_attacks_with_probability.csv` (6,555 rows, 12 columns).
Three columns are added to the cleaned dataset:

| Column | Type | Description |
|---|---|---|
| `attack_occurred` | int (0/1) | Binary label: 1 = confirmed attack, 0 = attempt only |
| `attack_probability_pct` | float | Model probability of confirmed attack, scaled 0–100 |
| `risk_band` | str | Categorical risk tier based on probability |

**Risk band thresholds:**

| Band | Probability Range |
|---|---|
| Low | 0% to < 25% |
| Moderate | 25% to < 50% |
| High | 50% to < 75% |
| Critical | 75% to 100% |

---

## 8. Dashboard

The Streamlit dashboard (`gui.py`) provides five tabs:

| Tab | Content |
|---|---|
| Dataset Overview | Data table, download button, attack type and vessel status charts |
| Attack Patterns | Geographic scatter map, temporal trend, seasonal heatmap, regional breakdown |
| Regression Model | CV R² distribution, MAE/RMSE per fold, feature importances, predicted vs. actual |
| Classification Model | AUC-ROC per fold, accuracy, Brier score, confusion matrix, feature importances |
| Probability Analysis | Probability histogram, risk band distribution, top-10 highest-risk incidents, download |

Sidebar filters (region, year range, risk band, minimum probability) apply across all tabs.

---

## 9. Testing

The test suite contains **100 unit tests** across four files, covering every component of
the pipeline with both structural and logical assertions.

```bash
source venv/bin/activate
pytest tests/ -v
```

Expected result: `100 passed`.

| File | Tests | Coverage |
|---|---|---|
| `tests/test_data_loading.py` | 23 | `detect_outliers_iqr`, `load_data`, `preprocess_data` |
| `tests/test_feature_engineering.py` | 29 | `engineer_features` (all 12 features) |
| `tests/test_pipeline.py` | 22 | `prepare_X_y`, `build_pipeline`, `build_classification_pipeline` |
| `tests/test_model_training.py` | 26 | `run_kfold_cv`, `run_classification_cv`, `fit_final_model`, `generate_attack_probability_column` |

---

## 10. Project Structure

```
zain_prob/
├── app.py                      # Core ML pipeline
├── gui.py                      # Streamlit dashboard
├── pirate_attacks_clean.csv    # Input dataset
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_feature_engineering.py
│   ├── test_pipeline.py
│   └── test_model_training.py
│
└── documentation/
    ├── 01_overview.md
    ├── 02_data_loading_and_preprocessing.md
    ├── 03_feature_engineering.md
    ├── 04_regression_model.md
    ├── 05_classification_model.md
    ├── 06_gui_dashboard.md
    └── 07_test_suite.md
```

---

## 11. Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| `numpy` | 1.24.0 | Numeric computation |
| `pandas` | 2.0.0 | Data manipulation |
| `scikit-learn` | 1.3.0 | ML pipelines, models, metrics |
| `plotly` | 5.18.0 | Interactive charts in the dashboard |
| `streamlit` | 1.30.0 | Web dashboard framework |
| `pytest` | (any) | Test runner |

Install all runtime dependencies:

```bash
pip install -r requirements.txt
```
