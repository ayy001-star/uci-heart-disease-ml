# UCI Heart Disease — ML Portfolio Project

Predicting presence of heart disease using the UCI Heart Disease dataset (Cleveland subset), comparing logistic regression and random forest classifiers.

## Project Structure

The project is organized into three numbered notebooks, meant to be run in order:

1. **`01_eda_uci.ipynb`** — Exploratory data analysis. Examines missingness by data-collection site, checks for implausible values, and justifies restricting the modeling dataset to the Cleveland site (lowest missingness).
2. **`02_preprocessing_uci.ipynb`** — Cleans and encodes the data: filters to Cleveland, drops rows with missing values in key clinical fields, encodes binary/nominal/ordinal variables, and produces a stratified 80/20 train-test split.
3. **`03_modeling_uci.ipynb`** — Trains and evaluates logistic regression and random forest models. Includes cross-validation, hyperparameter tuning (`GridSearchCV` for RF), McNemar's test to statistically compare the two models, ROC/calibration curves, SHAP feature importance, and inferential statistics (coefficients, odds ratios, confidence intervals, VIF) via `statsmodels`.

## Reproducing

1. Download the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) and place `heart_disease_uci.csv` in the project root (not included in this repo).
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn statsmodels shap matplotlib
   ```
3. Run the notebooks in order: `01` → `02` → `03`. Notebook `02` writes processed train/test splits to a `processed/` folder, which `03` reads from.

## Methodology Notes

- **Preprocessing is encapsulated in a `Pipeline`** (`ColumnTransformer` + `StandardScaler` + classifier) rather than scaling data manually before cross-validation. An earlier version of this project scaled the full training set once before running `cross_val_score`, which leaked information from each fold's held-out data into the scaler's fit statistics. Wrapping scaling inside the pipeline ensures it's refit independently within each CV fold.
- Because `ColumnTransformer` with `remainder="passthrough"` reorders columns (transformed columns first, passthrough columns after), feature names for coefficient/odds-ratio tables are pulled from `get_feature_names_out()` rather than assumed to match the original column order — an earlier version of the notebook had coefficients mislabeled as a result of this reordering.

## Model Comparison

Logistic regression and random forest are compared on test-set AUC-ROC and via McNemar's test on the test-set predictions, with additional diagnostics (calibration curve, SHAP values, VIF) to check model reliability and multicollinearity.


