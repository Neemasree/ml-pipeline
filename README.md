# AquaGuard AI — Explainable Predictive Maintenance for Water Distribution Pipelines

## Project Overview

AquaGuard AI is a standalone Machine Learning project that builds an explainable fault-detection
system for water distribution networks. It uses historical sensor data to predict
the likelihood of a fault condition (`fault_d7`) and provides SHAP-based explanations
for every prediction.

This is a **college ML project** — it is scientifically honest, reproducible, and
designed for future integration with a FastAPI backend.

---

## Project Objective

1. Load and inspect the water-distribution dataset
2. Clean and preprocess data (without fabricating or removing information carelessly)
3. Perform exploratory data analysis
4. Train and compare Random Forest and XGBoost classifiers
5. Select the best model based on F1-score and PR-AUC
6. Provide SHAP-based explainability for every prediction
7. Expose a reusable prediction API function ready for FastAPI integration

---

## Dataset

| Property | Value |
|---|---|
| Filename | `input_model_potenc_predXfault7_A.csv` |
| Rows | ~8,737 |
| Columns | 20 |
| Target | `fault_d7` (binary: 0=No Fault, 1=Fault) |
| Timestamp | `timestamp` (hourly records) |

### Feature Description

| Column | Description |
|---|---|
| `timestamp` | Hourly timestamp |
| `fault_d7` | **Target** — binary fault indicator |
| `4700008966_kW` | Power sensor reading (kW) |
| `4700008966_power_hour` | Power-hour accumulator |
| `4408411600_kW` | Second power sensor |
| `4408411600_power_hour` | Second power-hour accumulator |
| `4608927500_kvar_hour` | Reactive power sensor |
| `238045_ws_temp` | Weather station temperature |
| `279805_ws_vigor` | Weather station vigour/wind |
| `279804_ws_level` | Weather station level |
| `279625_ws_temp` | Second weather station temperature |
| `279623_ws_vigor` | Second weather station vigour |
| `319235_ws_temp` | Third weather station temperature |
| `279831_ws_temp` | Fourth weather station temperature |
| `326417_ws_temp` | Fifth weather station temperature |
| `326416_ws_vigor` | Fifth weather station vigour |
| `temp_site1_anomaly_score` | Pre-computed anomaly score (Site 1 temperature) |
| `sfc_temp_site1_anomaly_score` | Surface temperature anomaly score |
| `gw_lvl_site2_anomaly_score` | Groundwater level anomaly score (Site 2) |
| `gw_temp_site2_anomaly_score` | Groundwater temperature anomaly score (Site 2) |

---

## Data Preprocessing

### Cleaning Strategy

1. **Duplicate removal** — exact duplicates carry no new information
2. **Timestamp parsing** — converted to `datetime`, sorted chronologically
3. **Infinite values** — replaced with NaN (handled later by imputer)
4. **Missing target rows** — removed (cannot train without label)
5. **Target validation** — confirmed values are in {0, 1}
6. **Constant columns** — dropped (zero variance = no predictive signal)
7. **High-missing columns** — investigated and documented before removal
8. **Feature NaNs** — handled by `SimpleImputer(strategy="median")` fitted on training data only

### Feature Engineering

Time features extracted from `timestamp`:
- `hour` — time-of-day effects on water usage
- `day_of_week` — weekday vs weekend patterns
- `month` — seasonal effects
- `is_weekend` — binary flag

---

## EDA Highlights

- Target imbalance: investigate actual ratio in notebook output
- Anomaly score features show different distributions between classes
- Temporal trends visible in fault occurrence over time
- Correlation matrix available in `reports/figures/`

---

## Model Training

### Train/Test Split

**Temporal split** — 80% earliest records for training, 20% most recent for testing.
This mirrors real deployment and prevents data leakage from future observations.

### Random Forest

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | Stable predictions without excess compute |
| `max_depth` | None | Grows fully, regularized by min_samples_leaf |
| `min_samples_leaf` | 4 | Prevents over-fitting to tiny nodes |
| `class_weight` | balanced | Handles class imbalance automatically |

### XGBoost

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | Baseline |
| `max_depth` | 6 | XGBoost default |
| `learning_rate` | 0.05 | Conservative; reduces over-fitting |
| `subsample` | 0.8 | Row subsampling |
| `colsample_bytree` | 0.8 | Feature subsampling |
| `scale_pos_weight` | auto | Computed as neg/pos ratio |

---

## Model Evaluation

**Primary metrics** (preferred over accuracy for imbalanced classification):
- **F1-score** — balances precision and recall
- **PR-AUC** — Precision-Recall Area Under Curve; more informative than ROC-AUC for imbalanced data
- **Recall** — critical for fault detection (missed faults are costly)

Accuracy alone is **not** the selection criterion.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) explains each prediction:

- **Global view**: Which features matter most across all predictions?
- **Local view**: Why did the model predict fault/no-fault for this specific observation?

### Interpretation

| SHAP Value | Meaning |
|---|---|
| Positive (+) | Feature pushes prediction **toward fault (class 1)** |
| Negative (−) | Feature pushes prediction **away from fault (toward class 0)** |

---

## Risk Scoring

### Risk Score
```
risk_score = round(probability * 100)
```
- `risk_score = 82` means the model assigns 82% probability to fault_d7=1
- This is a **model-derived indicator**, not a direct physical measurement

### Risk Levels

| Level | Risk Score |
|---|---|
| Low | 0 – 34 |
| Medium | 35 – 59 |
| High | 60 – 79 |
| Critical | 80 – 100 |

---

## Pipeline Health Index (PHI)

PHI is a **derived project metric** — it is NOT measured directly by any sensor.

```
PHI = 100 - risk_score
```

- PHI = 100 → No predicted fault risk (healthy pipeline)
- PHI = 0   → Maximum predicted fault risk (critical)

---

## Repair Priority

| Priority | Risk Score | Recommendation |
|---|---|---|
| 1 (Immediate) | 80–100 | Immediate inspection recommended |
| 2 (Soon) | 60–79 | Schedule inspection soon |
| 3 (Monitor) | 35–59 | Monitor closely |
| 4 (Routine) | 0–34 | Routine maintenance |

---

## Time-to-Failure Limitation

> **Time-to-failure is NOT predicted in this project.**
>
> The dataset provides a binary fault indicator (`fault_d7 ∈ {0, 1}`).
> There are no failure timestamps, remaining-useful-life (RUL) labels,
> or sequential fault progression data.
>
> Fabricating TTF values would be scientifically dishonest.
>
> **Use instead:** `risk_level` + `priority_level` + `fault_probability`

---

## Project Structure

```
ml-pipeline/
├── data/
│   ├── raw/
│   │   └── input_model_potenc_predXfault7_A.csv
│   └── processed/
│       └── aquaguard_clean.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_shap_explainability.ipynb
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── final_model.pkl
│   └── preprocessing.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── reports/
│   ├── figures/
│   └── model_results/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Create Virtual Environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Dataset

Put the CSV file in:
```
data/raw/input_model_potenc_predXfault7_A.csv
```

---

## Running Notebooks

Run notebooks in order inside the `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

Order:
1. `01_data_understanding.ipynb` — inspect raw data
2. `02_data_cleaning.ipynb` — clean and save processed data
3. `03_eda.ipynb` — exploratory analysis
4. `04_model_training.ipynb` — train, evaluate, select model
5. `05_shap_explainability.ipynb` — SHAP explanations

---

## Loading the Final Model

```python
import sys
sys.path.insert(0, 'src')

from predict import load_model, load_preprocessing, full_prediction
import pandas as pd

# Load
model = load_model('models/final_model.pkl')
pp    = load_preprocessing('models/preprocessing.pkl')

# Predict on new data
X_new = pd.DataFrame([{
    'hour': 14,
    'month': 6,
    'temp_site1_anomaly_score': 45.2,
    # ... all features
}])

result = full_prediction(X_new, model=model, preprocessing_artifacts=pp)
print(result)
# {
#   "prediction": 1,
#   "probability": 0.82,
#   "risk_score": 82,
#   "risk_level": "Critical",
#   "phi": 18.0,
#   "priority": {...}
# }
```

---

## Future FastAPI Integration

```
React Frontend
      ↓
FastAPI Backend
      ↓
  (imports from src/)
      ↓
  final_model.pkl
  preprocessing.pkl
      ↓
  Prediction + SHAP
```

The `src/` module functions are designed to be imported directly:

```python
# In FastAPI route:
from src.predict import load_model, load_preprocessing, full_prediction
from src.explain import create_shap_explainer, explain_prediction
```

---

## Research Integrity Notes

- All metrics are computed from the actual trained model on real test data
- SHAP values are computed from the trained model — never fabricated
- Time-to-failure is not claimed (dataset does not support it)
- `risk_score` and `PHI` are clearly labeled as derived indicators
- Class imbalance is handled with `class_weight="balanced"` / `scale_pos_weight`
- No SMOTE is applied (class imbalance handled via model parameters)

---

*AquaGuard AI — College ML Project | Python 3.12 | scikit-learn | XGBoost | SHAP*
#   m l - p i p e l i n e  
 