---
title: Predict Credit Worthiness
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.38.2"
app_file: app.py
pinned: false
---

# Credit Worthiness Prediction
Predicts whether an applicant is a **good** or **bad** credit risk from
past financial/demographic data, using classical ML classifiers.

## Dataset

`data/CreditWorthiness.xlsx` — the German Credit dataset (1,000 applicants,
20 raw features + target `creditScore`). Classes are imbalanced: 700 "good",
300 "bad". Features include checking/savings balance, credit history, loan
amount & duration, employment duration, age, housing type, job type,
number of dependents, and more. No missing values.

## Feature Engineering

On top of the 20 raw fields, four derived features were added to better
capture repayment risk:

- **monthly_burden** — credit amount ÷ duration (repayment pressure per month)
- **age_group** — bucketed age (`<=25`, `26-35`, `36-45`, `46-60`, `60+`)
- **amt_per_dependent** — credit amount ÷ number of dependents (financial strain)
- **high_risk_combo** — flag for loans that are both above-median amount *and* above-median duration

All categorical columns were label-encoded. Logistic Regression uses
standardized features; the tree-based models use the raw encoded values
(scaling doesn't matter for trees).

## Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight="balanced"`, scaled features |
| Decision Tree | max_depth=6, min_samples_leaf=10, balanced |
| Random Forest | 300 trees, max_depth=8, min_samples_leaf=5, balanced |

`class_weight="balanced"` was used throughout since "bad" is the minority
(and more costly to miss) class.

## Results (held-out 25% test set, stratified)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** | 0.855 | 0.846 | 0.851 | **0.843** |
| Logistic Regression | 0.858 | 0.760 | 0.806 | 0.796 |
| Decision Tree | 0.881 | 0.674 | 0.764 | 0.736 |

*(Precision/recall/F1 above are weighted; see `outputs/model_comparison.csv`
for exact numbers and `classification_report` output in the console log for
the per-class breakdown.)*

**Random Forest is the best overall model** — highest ROC-AUC and the best
balance of precision/recall on the minority "bad" class (0.65 precision /
0.67 recall for "bad" applicants), which matters more in credit risk than
overall accuracy.

## Top Predictive Features (Random Forest importances)

1. `Cbal` — checking account balance
2. `monthly_burden` — engineered repayment-pressure feature
3. `Camt` — credit amount
4. `Cdur` — credit duration
5. `amt_per_dependent` — engineered strain feature
6. `Sbal` — savings balance
7. `age`

Two of the top 5 features are engineered ones, confirming they add real
predictive signal beyond the raw fields.

## Outputs

- `outputs/model_comparison.csv` — full metrics table
- `outputs/feature_importance.csv` / `.png` — Random Forest feature ranking
- `outputs/roc_curves.png` — ROC curve comparison across models
- `outputs/confusion_matrices.png` — confusion matrix per model
- `outputs/metric_comparison.png` — bar chart of Precision/Recall/F1/ROC-AUC
- `outputs/summary.json` — machine-readable summary
- `models/best_model_random_forest.joblib` — trained best model
- `models/scaler.joblib`, `models/label_encoders.joblib` — preprocessing artifacts (needed to score new applicants)

## Running It

```bash
pip install pandas scikit-learn matplotlib seaborn joblib openpyxl
python credit_scoring_pipeline.py
```

## Possible Next Steps

- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- Try gradient boosting (XGBoost/LightGBM) for a likely accuracy bump
- Calibrate probabilities (`CalibratedClassifierCV`) if the model will
  drive actual lending decisions, since raw RF probabilities aren't
  well-calibrated
- SHAP values for per-applicant explainability instead of just global
  feature importance
