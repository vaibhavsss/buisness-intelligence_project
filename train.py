#!/usr/bin/env python3
"""
Employee Attrition Prediction Model Trainer
Trains and tunes Logistic Regression & Random Forest models with proper preprocessing.
"""

# Top-level imports
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer categorical and numeric columns by dtype."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return categorical_cols, numeric_cols


def build_preprocess_pipeline(
    categorical_cols: List[str], numeric_cols: List[str]
) -> ColumnTransformer:
    """Build preprocessing pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")  # Nice column names in pipelines
    return preprocessor


# function evaluate_model()
# Fix: use X and y consistently inside the function to avoid NameError
def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, label_classes: List[str]) -> dict:
    """Evaluate a pipeline on validation or test data and return metrics."""
    y_pred = pipe.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="binary" if len(label_classes) == 2 else "weighted", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="binary" if len(label_classes) == 2 else "weighted", zero_division=0)),
        "f1": float(f1_score(y, y_pred, average="binary" if len(label_classes) == 2 else "weighted", zero_division=0)),
    }

    # Compute ROC-AUC if binary and proba available
    if len(label_classes) == 2 and hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X)
            metrics["roc_auc"] = float(roc_auc_score(y, y_proba[:, 1]))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def tune_candidate(
    name: str,
    base_pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    score_key: str,
    n_iter: int,
    random_state: int,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Hyperparameter tuning using RandomizedSearchCV."""
    if name == "logistic_regression":
        param_distributions = {
            "model__C": loguniform(1e-3, 1e3),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [1000],
        }
    elif name == "random_forest":
        param_distributions = {
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
        }
    else:
        raise ValueError(f"Unknown model: {name}")

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=score_key,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def plot_reports(final_pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, label_classes: List[str], out_dir: str, prefix: str = "eval"):
    """
    Generate and save confusion matrix, ROC (binary), PR curve, and top feature importance plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Predictions and probabilities
    y_pred = final_pipe.predict(X)
    y_proba = None
    if hasattr(final_pipe, "predict_proba"):
        try:
            y_proba = final_pipe.predict_proba(X)
        except Exception:
            y_proba = None

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_classes, yticklabels=label_classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()

    # ROC curve (binary only)
    if y_proba is not None and len(label_classes) == 2:
        try:
            RocCurveDisplay.from_predictions(y_true=y, y_pred=y_proba[:, 1])
            plt.title("ROC Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve.png"))
            plt.close()
        except Exception:
            pass

    # Precision-Recall curve
    if y_proba is not None and len(label_classes) == 2:
        try:
            PrecisionRecallDisplay.from_predictions(y_true=y, y_pred=y_proba[:, 1])
            plt.title("Precision-Recall Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png"))
            plt.close()
        except Exception:
            pass

    # Feature importance (if available)
    try:
        preprocess = final_pipe.named_steps["preprocess"]
        feature_names = preprocess.get_feature_names_out()
        model = final_pipe.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-20:]  # top 20
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_idx)), importances[top_idx], color="steelblue")
            plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
            plt.xlabel("Importance")
            plt.title("Top Feature Importances (Model)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_feature_importances.png"))
            plt.close()
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            # Aggregate coefficients for multiclass as mean absolute value
            if coefs.ndim == 2:
                agg = np.mean(np.abs(coefs), axis=0)
            else:
                agg = np.abs(coefs)
            top_idx = np.argsort(agg)[-20:]
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_idx)), agg[top_idx], color="darkorange")
            plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
            plt.xlabel("|Coefficient| (aggregated)")
            plt.title("Top Features by Coefficient Magnitude (Model)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_feature_coefficients.png"))
            plt.close()
    except Exception:
        # Silently skip feature importance plots if extraction fails
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Employee Attrition Prediction Model"
    )
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--test", required=False, help="Path to test CSV (with labels)")
    parser.add_argument("--target", default="Attrition", help="Target column name")
    parser.add_argument(
        "--output-model",
        default="models/attrition_model.joblib",
        help="Path to save the final model",
    )
    parser.add_argument(
        "--n-iter", type=int, default=30, help="Number of RandomizedSearch iterations"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(args.train)
    if args.target not in train_df.columns:
        raise ValueError(
            f"Target '{args.target}' not in training data. Columns: {list(train_df.columns)}"
        )

    test_df = None
    if args.test:
        test_df = pd.read_csv(args.test)

    # Prepare features and target
    X_full = train_df.drop(columns=[args.target])
    y_raw = train_df[args.target].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    label_classes = le.classes_.tolist()

    if len(label_classes) < 2:
        raise ValueError("Target must have at least 2 classes.")

    # Infer column types
    cat_cols, num_cols = infer_feature_types(X_full)
    print(f"Detected {len(cat_cols)} categorical and {len(num_cols)} numeric features.")

    # Preprocessing pipeline
    preprocessor = build_preprocess_pipeline(cat_cols, num_cols)

    # Model candidates
    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", random_state=args.random_state
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=args.random_state, n_jobs=-1
        ),
    }

    # CV and scoring
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    score_key = "roc_auc" if len(label_classes) == 2 else "f1_weighted"

    # Tuning
    best_estimator = None
    best_score = -np.inf
    best_name = None
    tuning_results = {}

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

        print(f"\nTuning {name.replace('_', ' ').title()}...")
        estimator, cv_score, params = tune_candidate(
            name=name,
            base_pipe=pipe,
            X=X_full,
            y=y,
            cv=cv,
            score_key=score_key,
            n_iter=args.n_iter,
            random_state=args.random_state,
        )

        tuning_results[name] = {"cv_score": cv_score, "best_params": params}

        if cv_score > best_score:
            best_score = cv_score
            best_estimator = estimator
            best_name = name

    if best_estimator is None:
        raise RuntimeError("Model selection failed.")

    # Refit best model on full training data
    print(f"\nSelected: {best_name.replace('_', ' ').title()} (CV {score_key}: {best_score:.4f})")
    best_estimator.fit(X_full, y)

    # Evaluation dictionary
    results = {
        "selected_model": best_name,
        "selection_metric": score_key,
        "cv_best_score": best_score,
        "best_params": tuning_results[best_name]["best_params"],
        "all_tuning_results": tuning_results,
    }

    # Test set evaluation (if provided and has labels)
    if test_df is not None and args.target in test_df.columns:
        X_test = test_df.drop(columns=[args.target]).reindex(columns=X_full.columns, fill_value=0)
        y_test_raw = test_df[args.target].astype(str)
        y_test = le.transform(y_test_raw)
        results["test_metrics"] = evaluate_model(best_estimator, X_test, y_test, label_classes)
        print("Test set metrics:", results["test_metrics"])

    # Save everything
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_estimator, output_path)
    print(f"Model saved to: {output_path}")

    # Save metadata
    meta_dir = output_path.parent
    with open(meta_dir / "model_classes.json", "w") as f:
        json.dump({"classes": label_classes, "target": args.target}, f, indent=2)

    with open(meta_dir / "features.json", "w") as f:
        json.dump(
            {
                "features": X_full.columns.tolist(),
                "categorical": cat_cols,
                "numeric": num_cols,
            },
            f,
            indent=2,
        )

    with open(meta_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()