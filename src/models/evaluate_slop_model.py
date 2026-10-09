import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CURRENT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = CURRENT_DIR.parent / "features"
sys.path.append(str(FEATURES_DIR))

from slop_features import calculate_slop_features


INPUT_PATH = "data/raw/slop_annotation_set.csv"
MODEL_PATH = "models/slop_detector.pkl"


def build_feature_dataset(df):
    rows = []

    for text in df["text"]:
        rows.append(calculate_slop_features(text))

    return pd.DataFrame(rows)


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    df["slop_label"] = pd.to_numeric(
        df["slop_label"],
        errors="coerce",
    )

    df = df.dropna(subset=["slop_label"]).copy()
    df["slop_label"] = df["slop_label"].astype(int)

    print(f"Samples: {len(df)}")
    print()

    print("Loading trained model...")
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    feature_names = saved["features"]

    print("Extracting features...")
    X = build_feature_dataset(df)
    X = X[feature_names]

    y = df["slop_label"]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df["predicted_slop"] = predictions
    df["slop_probability"] = probabilities

    print()
    print("=" * 70)
    print("FULL DATASET SLOP ML EVALUATION")
    print("=" * 70)

    print()
    print("Accuracy:")
    print(f"{accuracy_score(y, predictions):.4f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Not Slop", "Slop"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

    errors = df[
        df["slop_label"] != df["predicted_slop"]
    ]

    print()
    print("=" * 70)
    print("MISCLASSIFIED EXAMPLES")
    print("=" * 70)

    print(f"Errors: {len(errors)}")

    for _, row in errors.iterrows():
        actual = "SLOP" if row["slop_label"] == 1 else "NOT SLOP"
        predicted = (
            "SLOP"
            if row["predicted_slop"] == 1
            else "NOT SLOP"
        )

        print("-" * 70)
        print(f"ID: {row['id']}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print(f"Probability: {row['slop_probability']:.4f}")
        print()
        print(row["text"][:500])

    print("=" * 70)


if __name__ == "__main__":
    main()