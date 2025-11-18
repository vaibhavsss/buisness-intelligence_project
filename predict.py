#!/usr/bin/env python3
# top-level imports
import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def load_artifacts(model_path: str):
    """Load trained pipeline, features, and classes sidecars."""
    pipe = joblib.load(model_path)
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
    features_file = os.path.join(model_dir, "features.json")
    classes_file = os.path.join(model_dir, "model_classes.json")

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Missing features.json at {features_file}")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Missing model_classes.json at {classes_file}")

    with open(features_file, "r") as f:
        features_payload = json.load(f)
    with open(classes_file, "r") as f:
        classes_payload = json.load(f)

    features: List[str] = features_payload["features"]
    classes: List[str] = classes_payload["classes"]

    return pipe, features, classes


def main():
    parser = argparse.ArgumentParser(description="Generate attrition predictions from a trained model.")
    parser.add_argument("--model", required=True, help="Path to trained model pipeline (joblib).")
    parser.add_argument("--data", required=True, help="Path to CSV with employee features.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path for predictions.")
    parser.add_argument("--drop-target", default="Attrition", help="If present in input data, drop this target column. Default: Attrition")
    args = parser.parse_args()

    pipe, features, classes = load_artifacts(args.model)

    df = pd.read_csv(args.data)

    # Drop target column if present in input
    if args.drop_target in df.columns:
        df = df.drop(columns=[args.drop_target])

    # Align columns with training features (missing columns will be NaN)
    X = df.reindex(columns=features)

    # Predict encoded labels
    y_pred_enc = pipe.predict(X)
    # Map back to original labels
    # y_pred_enc values are indices into classes list if LabelEncoder was used in training
    # Ensure integer dtype
    y_pred_enc = np.array(y_pred_enc, dtype=int)
    y_pred_labels = [classes[i] for i in y_pred_enc]

    # Probabilities per class (if available)
    proba_df = pd.DataFrame(index=df.index)
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X)  # shape: (n_samples, n_classes)
            for i, cls in enumerate(classes):
                proba_df[f"proba_{cls}"] = y_proba[:, i]
        except Exception:
            # If predict_proba fails, skip probabilities
            pass

    out_df = pd.DataFrame({"prediction": y_pred_labels})
    if not proba_df.empty:
        out_df = pd.concat([out_df, proba_df], axis=1)

    # Save predictions
    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # Create and save pie chart of predicted class distribution
    counts = out_df["prediction"].value_counts()
    pie_filename = os.path.splitext(os.path.basename(args.output))[0] + "_pie.png"
    pie_path = os.path.join(out_dir, pie_filename)
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Predicted Attrition Distribution")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close()

    print(f"Predictions saved to: {args.output}")
    if not proba_df.empty:
        print("Included per-class probabilities:", list(proba_df.columns))
    print(f"Saved pie chart to: {pie_path}")


if __name__ == "__main__":
    main()