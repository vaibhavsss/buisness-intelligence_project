Employee Attrition Prediction

A clean, production-ready machine learning pipeline for predicting employee attrition using tabular HR data. Built entirely with scikit-learn — no manual feature engineering required.

Overview
This project trains and evaluates an Employee Attrition prediction model using separate training and test CSV files. It includes:
- Preprocessing for numeric and categorical features
- Model tuning (Logistic Regression and Random Forest) with cross-validation
- Evaluation metrics and saved artifacts
- Prediction script with per-class probabilities
- Automatic graphs: confusion matrix, ROC/PR curves (binary), feature importance, and a pie chart for predicted class distribution

Requirements
- Python 3.9+ (works on macOS)
- Install dependencies:
  - pip install -r requirements.txt

Data Requirements
- CSV files with feature columns and a target column (default: Attrition)
- Labels can be any strings (e.g., Yes/No, Left/Stayed); they’re recorded and used consistently
- If the test CSV lacks the target column, use the prediction script instead of test evaluation

Quick Start
- Train (with tuning and evaluation on test set if it has the target):
  - python3 train.py --train train.csv --test test.csv --target Attrition --output-model models/attrition_model.joblib --n-iter 20
- Predict (unlabeled test):
  - python3 predict.py --model models/attrition_model.joblib --data test.csv --output predictions.csv

Small-data setup for older Macs
- Create smaller CSVs to reduce memory/CPU load:
  - head -n 1001 "train.csv" > "train_small.csv"     # first 1000 rows + header
  - head -n 501 "test.csv" > "test_small.csv"         # first 500 rows + header
- Train with fewer tuning iterations:
  - python3 train.py --train train_small.csv --test test_small.csv --target Attrition --output-model models/attrition_model.joblib --n-iter 5

Training Details
- Preprocessing:
  - Numeric: median imputation + standard scaling
  - Categorical: most-frequent imputation + one-hot encoding (handle_unknown="ignore")
- Models tuned with 5-fold Stratified CV:
  - Logistic Regression (C, solver)
  - Random Forest (n_estimators, depth, splits, leaf size, max_features)
- Selection metric:
  - Binary target: ROC-AUC
  - Multiclass: F1-weighted
- Saved artifacts (in models/):
  - attrition_model.joblib (pipeline)
  - features.json (feature columns + types)
  - model_classes.json (original labels)
  - metrics.json (CV, params, evaluation)

Prediction Details
- Input CSV must contain the same feature columns used during training (missing columns handled via imputation)
- Output CSV includes:
  - prediction: original class label
  - proba_<class>: per-class probability (if available)

Graphs and Reports
- During training (if test or holdout labels are available), the following PNGs are saved under models/reports/:
  - test_confusion_matrix.png
  - test_roc_curve.png (binary only)
  - test_pr_curve.png (binary only)
  - test_feature_importances.png or test_feature_coefficients.png
- During prediction, a pie chart of predicted class distribution is saved next to the output CSV:
  - predictions_small_pie.png (for predictions_small.csv)
  - predictions_pie.png (for predictions.csv)

Troubleshooting
- FileNotFoundError: Use absolute paths if your directory contains spaces (e.g., "/Users/<you>/Documents/SEM 7/BIML/train.csv")
- Missing target column: pass the correct name via --target
- Slow training: reduce --n-iter, use train_small.csv/test_small.csv
- SciPy install issues: ensure pip, wheel, and setuptools are updated (pip install --upgrade pip setuptools wheel)

Project Structure
- README.md
- train.py
- predict.py
- requirements.txt
- models/
  - attrition_model.joblib
  - features.json
  - model_classes.json
  - metrics.json
  - reports/ (plots saved here)
- train.csv / test.csv (your datasets)
- train_small.csv / test_small.csv (optional small subsets)
- predictions.csv / predictions_small.csv (generated outputs)
- predictions_pie.png / predictions_small_pie.png (pie charts)

Test Results (latest run)
- Dataset: train_small.csv / test_small.csv
- Selected model: Logistic Regression
- Metrics on test set:
  - Accuracy: 0.758
  - Precision: 0.756198347107438
  - Recall: 0.746938775510204
  - F1: 0.7515400410677618
  - ROC-AUC: 0.8365266106442577

Maintainer
- VesmorianX 