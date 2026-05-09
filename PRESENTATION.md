# Predictive Modeling of US Used Vehicle Prices

ENGR422 Applied Machine Learning Course Project

Eren Acar Başaran (83179) and Ahmet Aybars Pektaş (91687)

---

## Slide 1 Title

Predictive Modeling of US Used Vehicle Prices.

ENGR422 Applied Machine Learning.

Eren Acar Başaran and Ahmet Aybars Pektaş.

[insert university and course logo]

---

## Slide 2 Problem and motivation

The second hand car market on Craigslist has a lot of noise. Prices are not standard. Some listings are spam. Some are missing data.

We want to build a model that predicts the price of a used car from its features. The features are year, mileage, brand, model, condition, fuel type, transmission, and others.

A good model helps buyers and sellers know fair prices. It also shows which features matter most for the price.

[insert one image of a Craigslist car listing or used car market photo]

---

## Slide 3 Dataset

Source dataset is Craigslist Vehicles from Kaggle.

It has 426880 rows and 26 columns.

Each row is one car listing. The target is the price column.

The dataset has many missing values. Some columns like county are 100 percent empty. Other columns like cylinders are 41 percent empty.

We use only car and truck listings. Boats, motorcycles, and RVs are out of scope.

[insert table or screenshot showing the first 5 rows of the dataset]

---

## Slide 4 Pipeline overview

Our pipeline has four stages.

1. Data foundations. Load the data and run exploratory data analysis.

2. Preprocessing. Clean outliers, handle missing values, and encode categorical features.

3. Model implementation. Train Linear Regression, Random Forest, and XGBoost.

4. Evaluation. Compare the three models on the held out test set.

[insert pipeline diagram showing the four stages from raw data to final model]

---

## Slide 5 EDA highlights

We found three big patterns in the data.

Price has a long tail. Most cars cost between 5000 and 30000 dollars. A few are listed at 0 dollars or 999999 dollars. These are spam.

Year and mileage are the strongest predictors. Newer cars and lower mileage cars have higher prices.

The relationship between price and year is not linear. New cars lose value fast. Old cars have a flat price.

[insert price distribution histogram from notebook 01]

[insert price vs year scatter plot from notebook 01]

---

## Slide 6 Preprocessing decisions

We made four main preprocessing choices.

We dropped columns that have too many missing values. We dropped county at 100 percent missing, size at 71 percent missing, and cylinders at 41 percent missing.

We removed price outliers. We kept only listings between 500 and 100000 dollars.

We capped mileage at 300000 miles to remove unrealistic values.

We split the data into 80 percent train and 20 percent test before any imputation, so test data does not leak into the training stage.

[insert before and after price distribution comparison]

---

## Slide 7 Encoding strategy

The dataset has both numeric and categorical features. We treat each type differently.

For numeric features like year and mileage, we use scaling for the linear model and we leave them raw for tree models.

For low cardinality categorical features like fuel and transmission, we use one hot encoding.

For high cardinality features like manufacturer and model, we use mean target encoding. This replaces each category with the mean price of that category in the training data.

We also wrote five custom imputers that fill missing values smartly. For example a missing condition gets the most common condition for that mileage range.

[insert flow chart of the preprocessing column transformer]

---

## Slide 8 Three models

We compared three model families.

Linear Regression is our baseline. We tried plain OLS, Ridge with L2 penalty, and Lasso with L1 penalty. We tuned alpha with GridSearchCV.

Random Forest is the bagging ensemble. Aybars tuned it with RandomizedSearchCV over 5 hyperparameters.

XGBoost is the gradient boosting ensemble. We tuned it with RandomizedSearchCV over 8 hyperparameters with industry standard ranges.

All three models share the same preprocessing pipeline. They differ only in the estimator and small numeric handling.

[insert side by side icons or diagrams of linear vs forest vs boosting]

---

## Slide 9 Cross validation and tuning

We use 5 fold cross validation on the training set for all model comparisons.

We never touch the test set during training or hyperparameter selection. The test set is reserved for the final report only.

For tuning, Linear uses GridSearchCV because it has only one alpha to tune. Random Forest and XGBoost use RandomizedSearchCV because their search space is too big for a full grid.

For XGBoost we also tested early stopping as a second tuning method. Section 14 covers this.

The primary metric is Mean Absolute Error in dollars. This is easy to interpret and it is robust to extreme values.

[insert k fold cross validation diagram]

---

## Slide 10 Headline results

This is the main result of the project.

| Model | Test MAE | Test R squared |
|---|---|---|
| Linear Regression Ridge | 5378 dollars | 0.65 |
| Random Forest | 2154 dollars | 0.89 |
| XGBoost random search | 2494 dollars | 0.89 |
| XGBoost early stopping | 2175 dollars | 0.90 |

XGBoost with early stopping wins on RMSE, R squared, and MAPE. Random Forest wins on MAE by only 20 dollars which is statistical noise.

Both ensembles cut the linear baseline error by about 60 percent. This confirms our hypothesis that tree ensembles are needed for this kind of complex tabular data.

[insert MAE bar chart from notebook 04 section 4.2]

---

## Slide 11 Predicted vs actual

This plot shows how close each model gets to the true price.

The dashed line is the perfect prediction line. Closer to the line is better.

Linear regression has wide spread. The model under predicts expensive cars and over predicts cheap ones.

Random Forest and XGBoost are tightly packed around the line. They follow the diagonal closely up to about 50000 dollars.

Above 50000 dollars all models start to fail because there are not many luxury car listings to learn from.

[insert predicted vs actual scatter for Random Forest from notebook 04 section 4.3]

[insert predicted vs actual scatter for XGBoost from notebook 04 section 4.3]

---

## Slide 12 Drivers of price

What features actually drive price in our models.

Both Random Forest and XGBoost agree on the top features.

Year is the strongest single feature. Newer cars are worth more.

Mileage is second. More miles drop the price quickly.

Manufacturer mean price is third. Brand prestige matters a lot.

Model is fourth. Specific model lines have strong reputations.

Fuel type, drive, and transmission are smaller signals.

[insert top 15 feature importance plot from notebook 04 section 4.4]

---

## Slide 13 Where models fail

Our error analysis from notebook 04 section 4.5 shows three failure patterns.

Luxury cars over 50000 dollars have an average error of 8200 dollars. This is 5 times worse than mid range cars. There are very few luxury listings to learn from.

Common brands like Hyundai and Honda have low errors around 1200 to 1400 dollars. Premium brands like Mercedes have higher errors around 2700 dollars.

The worst predictions are mostly bad data. Some classic cars from 1953 are listed at 8000 dollars and a 2020 Ford F250 is listed at 1100 dollars. These are data quality issues, not model failures.

[insert MAE by price band bar chart from notebook 04 section 4.5]

[insert MAE by manufacturer bar chart from notebook 04 section 4.5]

---

## Slide 14 Ablation experiments

We tested two design questions on the held out test set.

Question 1, feature selection. We refit XGBoost on the top half of features by importance. Result: it hurt the model. The reduced model lost 183 dollars of MAE. Verdict: keep all features.

Question 2, tuning method. We trained XGBoost two ways with the same hyperparameters. Random search picked a fixed number of trees from the 100 to 800 range. Early stopping let the model add trees up to a 2000 ceiling and stop when validation MAE stopped improving. Result: early stopping won by 319 dollars of MAE. The reason: random search's tree count range was too narrow, so it never tested values above 800. Early stopping let the model decide.

Lesson: early stopping is more robust because it does not need a guess for the upper bound on tree count.

[insert bar chart comparing full XGBoost vs reduced features vs early stopping on test MAE]

---

## Slide 15 Trade offs

| Dimension | Linear Ridge | Random Forest | XGBoost early stopping |
|---|---|---|---|
| Test MAE | 5378 dollars | 2154 dollars | 2175 dollars |
| Test R squared | 0.65 | 0.89 | 0.90 |
| Model size on disk | 2.4 megabytes | 670 megabytes | 38 megabytes |
| Inference speed | very fast | slow | fast |
| Interpretability | high, signed coefficients | medium, gain importance | medium, gain importance |
| Hyperparameter sensitivity | low, one knob | medium, five knobs | high, eight knobs |

XGBoost with early stopping wins on R squared, RMSE, and MAPE. Random Forest is essentially tied on MAE but is 18 times bigger on disk and slower at inference. Linear is small and fast but much less accurate.

[insert side by side bar chart of MAE and model size]

---

## Slide 16 Recommendation

For production use we recommend XGBoost with early stopping. It has the highest R squared, the lowest RMSE, and the lowest MAPE on the held out test set. It is also smaller and faster than Random Forest.

Random Forest stays as a close alternative. Its MAE is essentially tied with XGBoost early stopping, only 20 dollars apart.

For interpretability we recommend keeping the Linear Ridge model as a fallback. Its coefficients give a clear dollar per feature explanation that the trees cannot match.

[insert summary infographic showing the three model recommendations]

---

## Slide 17 Limitations and future work

Our model has four main limits.

The data comes from one source, Craigslist. It may not generalize to dealership listings or other markets.

We did not use the free text description column. Adding text features could improve the model.

Geographic effects are coarse. We use state but not region or zip code. Local market effects are missed.

The luxury car segment over 50000 dollars has high error. More data on luxury cars or a separate model for that segment would help.

[insert limitations icon set or simple bullet list image]

---

## Slide 18 Thank you and questions

Thank you for listening.

Questions and discussion.

Repository link.

Contact emails.

[insert team photo or project logo]
