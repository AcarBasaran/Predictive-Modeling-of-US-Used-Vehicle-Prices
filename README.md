# Predictive Modeling of US Used Vehicle Prices

ENGR422 Applied Machine Learning

Eren Acar Başaran 83179
Ahmet Aybars Pektaş 91687

## What we do

We predict prices of used cars in the US. We use the Craigslist Vehicles dataset from Kaggle. We train four models. We compare them. The models are Linear Regression Random Forest XGBoost and LightGBM. We evaluate them with MAE RMSE R squared and MAPE.

## How to run

Clone the repo. Download vehicles.csv from Kaggle. Put it in the data folder. Install the packages.

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

Run the notebooks in order. First 01. Then 02. Then 03a 03b 03c 03d in any order. Then 04.

## Notebooks

| Notebook | What it does |
| --- | --- |
| 01_eda | Look at the data |
| 02_preprocessing | Clean the data. Build the pipeline |
| 03a_baseline_linear_regression | Train Ridge and Lasso |
| 03b_random_forest | Train Random Forest. Tune it |
| 03c_xgboost | Train XGBoost. Tune it |
| 03d_lightgbm | Train LightGBM. Tune it |
| 04_evaluation | Compare all models on the test set |

## Results

XGBoost is the best. Test MAE is around 2000 dollars. R squared is around 0.90.

## Folders

data/ ignored by git. You must download vehicles.csv yourself.
models/ ignored by git. The pkl files are made when you run the notebooks.
notebooks/ all the analysis and the models.
src/ shared helper code and custom preprocessing classes.

## Tech stack

Python 3.11. Pandas. NumPy. Matplotlib. Seaborn. Scikit-Learn. XGBoost. LightGBM. Joblib.
