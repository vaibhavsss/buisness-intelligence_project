Employee Attrition Prediction

Overview:
This project provides scripts to train an attrition prediction model and generate predictions on new data. It expects a CSV dataset containing employee features and a target column (default: "Attrition" with values like "Yes"/"No").

Features:
- Preprocessing with imputation, one-hot encoding for categoricals, and scaling for numerics
- Trains Logistic Regression and Random Forest, picks the best based on validation metrics
- Saves the trained pipeline, feature names, and label classes for consistent predictions
- Outputs evaluation metrics and predictions with per-class probabilities

Prerequisites:
- Python 3.9+ recommended
- Install dependencies

Setup:
1) Install dependencies:
   pip install -r requirements.txt

Training:
- Place your dataset as CSV (e.g., data/employee_attrition.csv).
- Ensure the target column exists (default: Attrition). You can override with --target.

Example:
  python3 train.py --data data/employee_attrition.csv --target Attrition --output-model models/attrition_model.joblib

Outputs:
- models/attrition_model.joblib: Trained pipeline (preprocessing + model)
- models/features.json: Ordered list of feature columns used during training
- models/model_classes.json: Original label classes in training order
- models/metrics.json: Validation metrics for selected model

Prediction:
Provide a CSV with the same feature columns (the script will reindex and handle missing columns).
Example:
  python3 predict.py --model models/attrition_model.joblib --data data/new_employee_data.csv --output predictions.csv

Prediction output:
- predictions.csv with:
  - prediction: Original label (e.g., "Yes"/"No")
  - proba_<class>: Probability of each class

Notes:
- If your target column name differs, pass it via --target.
- The pipeline is robust to missing values and unseen categories via SimpleImputer and OneHotEncoder(handle_unknown="ignore").
- Class imbalance is handled with class_weight="balanced" in the classifiers.


Test Results (latest run)
- Dataset: train_small.csv / test_small.csv
- Selected model: Logistic Regression
- Metrics on test set:
  - Accuracy: 0.758
  - Precision: 0.756198347107438
  - Recall: 0.746938775510204
  - F1: 0.7515400410677618
  - ROC-AUC: 0.8365266106442577