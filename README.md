# Home Credit Default Risk Modeling

This project predicts the probability that a Home Credit applicant will default. The modeling workflow is implemented in `baseline_metrics.R` and uses `application_train.csv` for development and `application_test.csv` for Kaggle scoring.

## Data and evaluation

`TARGET = 1` represents a default. The training data contain 307,511 applications, including 24,825 defaults (8.07%). Because the target is imbalanced, models are compared using cross-validated ROC AUC rather than accuracy. The Kaggle test file has no `TARGET` column, so it cannot be used to calculate accuracy or AUC.

## Baseline result

The majority-class classifier predicts `TARGET = 0` for every applicant.

| Model | Validation approach | Accuracy | ROC AUC |
| --- | --- | ---: | ---: |
| Majority class (`TARGET = 0`) | Training-set class distribution | 91.93% | 0.500 |

The high accuracy is caused by the low default rate and does not indicate useful ranking ability. Its AUC of 0.500 is the reference point that all predictive models should exceed.

## Models included in the notebook workflow

The reusable script compares the following models with stratified cross-validation:

| Model | Predictor set | Class-imbalance strategies |
| --- | --- | --- |
| Logistic regression | Numeric predictors | None, upsampling, downsampling, SMOTE |
| Logistic regression | All available predictors | None, upsampling, downsampling, SMOTE |
| Random forest | All available predictors | None, upsampling, downsampling, SMOTE |
| XGBoost | All available predictors | None, upsampling, downsampling, SMOTE |

Within every cross-validation fold, missing-value handling, category encoding, and any resampling occur using only the fold's training partition. The validation partition retains its original default rate, preventing data leakage and giving a fair out-of-sample AUC estimate.

## Cross-validation results

The model benchmark has not yet been run or saved in this workspace. Add the output of `benchmark$summary` here after running the notebook; do not select a final model until those results are available.

| Model | Imbalance method | Mean cross-validated AUC | AUC standard deviation | Status |
| --- | --- | ---: | ---: | --- |
| Logistic regression: numeric predictors | Pending | N/A | N/A | Not yet run |
| Logistic regression: all predictors | Pending | N/A | N/A | Not yet run |
| Random forest | Pending | N/A | N/A | Not yet run |
| XGBoost | Pending | N/A | N/A | Not yet run |

## Final-model selection

**Final model: pending cross-validation results.** The selected model will be the model-and-imbalance-strategy combination with the largest mean cross-validated ROC AUC. If two options are essentially tied, prefer the simpler and more stable option, using the fold-to-fold AUC standard deviation as a secondary check.

After selection, the notebook uses randomized hyperparameter search on a stratified 5,000-row subset with 3-fold cross-validation, then refits the selected workflow on the full labeled training data. The fitted model generates `TARGET` probabilities for `application_test.csv` and writes a Kaggle-ready submission file with `SK_ID_CURR` and `TARGET` columns.

## Run sequence

```r
source("baseline_metrics.R")

home_credit <- read.csv("application_train.csv", check.names = FALSE)

benchmark <- benchmark_models_cv(
  home_credit,
  target = "TARGET",
  id_columns = "SK_ID_CURR",
  v = 5
)

benchmark$summary
```

Update the Cross-validation results and Final-model selection sections with the resulting AUC values before creating a Kaggle submission.
