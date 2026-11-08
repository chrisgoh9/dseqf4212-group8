# DSE4212 / QF4212 Group 8: Data-Driven Credit Scoring Using Explainable Machine Learning

Project 1 of the DSE4212/QF4212 group project. We predict the probability that a loan applicant defaults, using the Statlog German Credit dataset, and compare three models: logistic regression, random forest and AdaBoost. Every model is judged on the bank's asymmetric cost of errors, explained with SHAP, and checked for differences between demographic groups.

**Team:** Calista Wong Xuan, Goh Chee Wei Christopher, Crystal Li Sansan, Neoh Jing Herng, Keira Low Wei-Qi, Tan Shao Gjin

---

## 1. The problem in one paragraph

A bank does not know in advance whether a borrower will repay. Approving a borrower who later defaults costs much more than turning away one who would have repaid. The dataset documentation (`german.doc`) sets these costs:

|                  | Borrower actually good | Borrower actually bad |
|------------------|------------------------|-----------------------|
| **Bank approves** | 0                      | **5**                 |
| **Bank rejects**  | **1**                  | 0                     |

With these costs, rejecting is cheaper whenever the predicted probability of default `p` satisfies `5p > 1 - p`, i.e. **`p > 1/6`**. All models in this project therefore flag an applicant as high risk when `P(default) > 1/6`, not 0.5, and are compared on **cost per applicant** as well as AUC, precision, recall and accuracy.

---

## 2. Repository structure

```
dseqf4212-group8/
├── README.md
├── dseqf4212-group8.ipynb              # the whole project, parts (a) to (c), run top to bottom
└── data/
    ├── raw/
    │   ├── german.data                 # original UCI file: 1,000 rows, 20 attributes + label, no header
    │   └── german.doc                  # code book: attribute meanings, codes A11..A202, cost matrix
    └── processed/
        ├── german_clean.csv            # output of part (a): decoded labels, `default` target, `sex`, log columns
        └── german_clean_modified.csv   # output of part (b): + 15 engineered features + `split` column
```

The notebook writes both processed files, so they can be regenerated from `data/raw/german.data` at any time.

---

## 3. Dataset

- **Source:** [Statlog (German Credit Data), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Size:** 1,000 applicants, 20 attributes (7 numerical, 13 categorical), amounts in Deutsche Marks (DM)
- **Target:** 700 good (70%), 300 bad (30%). The raw label is 1 = good, 2 = bad; we recode it as `default = 1` for bad so that default is the positive class.
- **No income variable.** True loan-to-income and debt-to-income ratios cannot be built, so part (b) builds proxies from the instalment rate (instalment as a % of disposable income).
- **Sensitive attributes:** `personal_status_sex` (sex combined with marital status), `age` and `foreign_worker`. A separate `sex` column is derived in part (a) for the fairness analysis.

---

## 4. Notebook walkthrough

The notebook is organised by the parts of the project brief. Variables in each model section carry a suffix (`_lr`, `rf_`, `_ada`) so the sections do not overwrite each other.

### Part (a): Data cleaning and EDA (sections 1 to 8)

| Section | What it does |
|---|---|
| 1. Load and decode | Reads `german.data`, names the columns and maps codes (e.g. `A11`) to readable labels from `german.doc` |
| 2. Integrity checks | No missing cells, no duplicates, all codes valid. Flags 33 "unemployed" applicants whose job is management / self-employed |
| 3. Target and cost matrix | 70/30 class balance; derives the 1/6 cut-off from the 5:1 cost matrix |
| 4. "Hidden" missing values | No checking account, unknown savings and unknown property are kept as their own categories because they predict default |
| 5. Numeric variables | Skewness, `log1p` transforms, IQR outliers (all kept: they are real and higher-risk) |
| 6. Categorical variables | Identifies levels with fewer than 30 applicants |
| 7. Relationship with default | Default rate by level and by quintile; Spearman correlations |
| 8. Save | Writes `data/processed/german_clean.csv` |

Key findings: checking account status has the clearest link with default (49.3% for `< 0 DM` down to 11.7% for no account); default rises with loan duration and for the largest loans; young applicants (≤ 26) default more; duration and credit amount are strongly correlated (0.62).

### Part (b): Feature engineering (sections 9 to 13)

| Feature | Definition | Purpose |
|---|---|---|
| `monthly_payment` | `credit_amount / duration` | size of the repayment (no interest) |
| `dti` | `installment_rate` | debt-to-income proxy for this loan |
| `income_index` | `monthly_payment / installment_rate` | relative disposable income |
| `lti` | `credit_amount / income_index` (= `duration × installment_rate`) | loan-to-income proxy |
| `total_dti` | `installment_rate × existing_credits` | total debt burden proxy |
| `other_installments` | 1 if instalment plans at other banks or stores | debt outside this bank |
| `log_monthly_payment`, `log_income_index`, `log_lti` | `log1p` of the above | less skew for linear models |
| `emp_category` | self-employed / unemployed / < 1 / 1-4 / 4-7 / ≥ 7 yrs | separates likely self-employed from unemployed |
| `emp_years` | tenure as an ordinal score 0 to 4 | ordered version of `employment` |
| `purpose_grp` | rare purposes merged (every level ≥ 46 applicants) | stable estimates in CV folds |
| `age_group` | < 25, 25-34, 35-49, 50+ | non-linear age effect |
| `amount_bin` | credit-amount quintiles, edges fitted on the training rows | U-shaped amount effect |
| `split` | stratified 80/20, `random_state=42` | the same 800 / 200 split for every model |

Output: `data/processed/german_clean_modified.csv` (1,000 rows, 40 columns). No original column is changed or removed.

### Part (c): Models

All models are trained on the 800 training applicants, tuned with stratified 5-fold cross-validation ranked by ROC-AUC, and evaluated once on the 200 test applicants at the 1/6 cut-off.

**(i) Logistic regression.** Starts from 20 candidate variable groups (log duration and amount, `age_group`, original categoricals). Backward elimination drops `other_debtors`, `job`, `existing_credits` and `residence_since` (CV AUC 0.7715 → 0.7805). L2 strength tuned over 11 values of `C`; best `C = 0.0316` (CV AUC 0.7850). Explained with SHAP `LinearExplainer` (beeswarm and two waterfall plots).

**(ii) Random forest.** Three configurations: a 100-tree baseline on all 38 predictors, Model 1 (tuned, 38 predictors) and Model 2 (tuned, 21 original predictors, no log / grouped / engineered features). Tuned with `RandomizedSearchCV` (30 combinations). Model 2 is selected because it has the lowest cross-validated cost per applicant (0.530). Explained with SHAP `TreeExplainer`, plus partial dependence (PDP) and ICE plots for duration and credit amount.

**(iii) AdaBoost.** 26 features (65 after one-hot encoding). Grid search over number of trees, learning rate and tree depth; best is 400 stumps with learning rate 0.5 (CV AUC 0.796). Raw AdaBoost probabilities are bunched between 0.26 and 0.61, so they are rescaled with sigmoid calibration (`CalibratedClassifierCV`) before applying the 1/6 cut-off. Explained with built-in feature importance and SHAP (permutation explainer).

---

## 5. Results (test set, 200 applicants, cut-off 1/6)

| Model | AUC | Precision (bad) | Recall (bad) | Accuracy | TN | FP | FN | TP | Cost per applicant |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.783 | 0.350 | 0.917 | 0.465 | 38 | 102 | 5 | 55 | 0.635 |
| **Random forest (Model 2)** | **0.803** | 0.395 | **0.967** | 0.545 | 51 | 89 | 2 | 58 | **0.495** |
| AdaBoost (calibrated) | 0.799 | 0.397 | 0.867 | 0.565 | 61 | 79 | 8 | 52 | 0.595 |
| Approve everyone | – | – | 0 | 0.700 | 140 | 0 | 60 | 0 | 1.500 |
| Reject everyone | – | 0.300 | 1 | 0.300 | 0 | 140 | 0 | 60 | 0.700 |

All three models are far cheaper than approving everyone. Accuracy is low by design: the 1/6 cut-off deliberately accepts many false rejections (cost 1 each) to avoid approving defaulters (cost 5 each). Random forest Model 2 has the lowest cost, but the gap between models is a handful of applicants on a 200-row test set, so the ranking is not certain.

The most influential features across models are checking account status, loan duration, credit amount, savings and the instalment rate. All three models reproduce the counter-intuitive EDA pattern for credit history ("critical account / other credits" lowers predicted risk).

---

## 6. How to run

1. Install Python 3.10+ and the packages below:

   ```bash
   pip install numpy pandas "matplotlib>=3.9" scikit-learn shap jupyter
   ```

   `matplotlib >= 3.9` is needed for the `tick_labels` argument in the box plots.

2. Open the notebook from the repository root, so the relative paths `data/raw/...` and `data/processed/...` resolve:

   ```bash
   jupyter notebook dseqf4212-group8.ipynb
   ```

3. Run all cells top to bottom. Parts (a) and (b) regenerate both CSV files; each model section reloads `german_clean_modified.csv`, so a model section can also be run on its own after the imports in part (a).

**Run time.** Everything except the AdaBoost SHAP cell runs in a few minutes. The AdaBoost SHAP cell uses the permutation explainer (about 10 s per applicant, around 30 minutes for all 200). Set `N_EXPLAIN = 20` in that cell for a quick check.

**Reproducibility.** The split, cross-validation folds and models use `random_state=42`. Exact random forest search results can differ slightly across scikit-learn versions, because `RandomizedSearchCV` may sample different parameter combinations; the selected Model 2 settings are `n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight=None`.

---

## 7. Conventions

- `default = 1` means bad credit (the positive class); `default = 0` means good.
- The test rows (`split == "test"`) are used only for the final evaluation, never for selection or tuning.
- Decision rule everywhere: reject when `P(default) > 1/6`.
- Cost per applicant = `(5 × FN + 1 × FP) / n`.
- The affordability features (`dti`, `income_index`, `lti`, `total_dti`) are relative rankings built from a 1 to 4 code, not amounts in DM.
