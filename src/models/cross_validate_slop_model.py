import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CURRENT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = CURRENT_DIR.parent / "features"
sys.path.append(str(FEATURES_DIR))

from slop_features import calculate_slop_features


INPUT_PATH = "data/raw/slop_annotation_set.csv"


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

    print("Extracting features...")
    X = build_feature_dataset(df)
    y = df["slop_label"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print()
    print("Running 5-fold cross-validation...")

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
    )

    print()
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION")
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


if __name__ == "__main__":
    main()