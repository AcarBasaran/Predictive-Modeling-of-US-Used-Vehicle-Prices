# Predictive Modeling of US Used Vehicle Prices

**Course:** ENGR422 — Applied Machine Learning  
**Team:** Eren Acar Başaran (83179), Ahmet Aybars Pektaş (91687)

A machine learning regression pipeline that predicts second-hand vehicle prices in the US market using the [Craigslist Vehicles dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data). The project compares three models — Linear Regression (baseline), Random Forest, and XGBoost — and evaluates them with MAE, RMSE, R², and MAPE.

---

## Repository Structure

```
├── data/                   # Dataset directory (git-ignored)
│   └── README.md           # Download instructions for vehicles.csv
├── models/                 # Saved .pkl model files (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb              # Data cleaning & Scikit-Learn pipeline
│   ├── 03a_baseline_linear_regression.ipynb  # Baseline model
│   ├── 03b_random_forest.ipynb             # Ensemble model (tuned)
│   ├── 03c_xgboost.ipynb                   # Gradient boosting model (tuned)
│   └── 04_evaluation.ipynb                 # Final comparison & results
├── .gitignore
└── README.md
```


## Getting Started

### 1. Clone & switch branch

```bash
git clone <repo-url>

```

### 2. Download the dataset

Go to [Kaggle — Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data), download `vehicles.csv`, and place it in the `data/` folder.

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 4. Run notebooks in order

Execute the notebooks sequentially: `01` → `02` → `03a/03b/03c` (any order) → `04`.

---

## Project Pipeline

| Phase | Notebook | Description |
|-------|----------|-------------|
| WP1 — Data Foundations | `01_eda.ipynb` | Load data, visualize distributions, identify outliers |
| WP2 — Preprocessing | `02_preprocessing.ipynb` | Outlier removal, train-test split, imputation, encoding |
| WP3 — Modeling | `03a`, `03b`, `03c` | Train & tune each model independently |
| WP4 — Evaluation | `04_evaluation.ipynb` | Compare all models, generate final metrics table |

---

## Next Steps / TODO

### !! CRITICAL — Do These First
- [ ] **!! Download `vehicles.csv`** from Kaggle and place in `data/` — nothing works without this
- [ ] **!! Agree on the exact `random_state` value** to use across all notebooks so the train-test split is identical everywhere
- [ ] **!! Decide who owns which notebook** — assign 03a/03b/03c between Eren and Aybars to avoid duplicate work

### Data & EDA
- [ ] Complete `01_eda.ipynb` — load data, explore distributions, find outlier thresholds
- [ ] Decide which columns to drop (likely: `id`, `url`, `region_url`, `image_url`, `VIN`, `description`, `county`)
- [ ] Determine price filtering bounds (1st/99th percentiles from proposal)

### Preprocessing
- [ ] Complete `02_preprocessing.ipynb` — build the full Scikit-Learn pipeline
- [ ] **!! Split BEFORE fitting** any imputer/encoder to avoid data leakage
- [ ] Decide Target Encoding vs. One-Hot for each categorical column
- [ ] Export a reusable preprocessing function/pipeline that model notebooks can import

### Modeling
- [ ] Complete `03a_baseline_linear_regression.ipynb`
- [ ] Complete `03b_random_forest.ipynb` — tune with RandomizedSearchCV
- [ ] Complete `03c_xgboost.ipynb` — tune with RandomizedSearchCV, use early stopping
- [ ] Save each trained pipeline as `.pkl` in `models/`

### Evaluation & Delivery
- [ ] Complete `04_evaluation.ipynb` — load all 3 models, generate comparison table
- [ ] Create visualizations for the final presentation (predicted vs. actual, feature importances)
- [ ] Prepare D4.2: Final Project Presentation

---

## Metrics

All models are evaluated on the **test set** using:

| Metric | What it measures |
|--------|-----------------|
| **MAE** | Average absolute prediction error in dollars |
| **RMSE** | Penalizes large errors more heavily |
| **R²** | Proportion of variance explained (higher = better) |
| **MAPE** | Percentage-based error for interpretability |

---

## Tech Stack

- Python 3.11+
- Pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualization
- Scikit-Learn — preprocessing pipelines, Linear Regression, Random Forest
- XGBoost — gradient boosting
- Joblib — model serialization
