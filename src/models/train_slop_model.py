import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib


# Allow imports from src/features
CURRENT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = CURRENT_DIR.parent / "features"
sys.path.append(str(FEATURES_DIR))

from slop_features import calculate_slop_features


INPUT_PATH = "data/raw/slop_annotation_set.csv"
MODEL_PATH = "models/slop_detector.pkl"


def build_feature_dataset(df):
    feature_rows = []

    print("Extracting features...")

    for text in df["text"]:
        features = calculate_slop_features(text)
        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    return features_df


def train_slop_model():
    print("Loading annotated dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["slop_label"] = pd.to_numeric(
        df["slop_label"],
        errors="coerce"
    )

    df = df.dropna(subset=["slop_label"])

    df["slop_label"] = df["slop_label"].astype(int)

    print(f"Samples: {len(df)}")
    print()

    # Build feature matrix
    X = build_feature_dataset(df)
    y = df["slop_label"]

    print(f"Features: {X.shape[1]}")
    print()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    print("Training Slop model...")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print()
    print("=" * 70)
    print("SLOP ML MODEL EVALUATION")
    print("=" * 70)
    print()

    print("Accuracy:")
    print(f"{accuracy_score(y_test, y_pred):.4f}")
    print()

    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not Slop", "Slop"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("Top features:")
    print(importance.head(15).to_string(index=False))

    # Save model
    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
        },
        MODEL_PATH,
    )

    print()
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_slop_model()