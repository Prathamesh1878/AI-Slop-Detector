import pandas as pd

from slop_score import calculate_slop_score
from slop_features import calculate_slop_features
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


INPUT_PATH = "data/raw/slop_annotation_set.csv"


def evaluate_slop_score():
    print("Loading annotated dataset...")

    df = pd.read_csv(INPUT_PATH)

    # Make sure slop labels are numeric
    df["slop_label"] = pd.to_numeric(
        df["slop_label"],
        errors="coerce",
    )

    # Remove anything that has not been annotated
    df = df.dropna(subset=["slop_label"])

    df["slop_label"] = df["slop_label"].astype(int)

    print(f"Annotated samples: {len(df)}")
    print()
    print("Label distribution:")
    print(df["slop_label"].value_counts().sort_index())

    print()
    print("Calculating slop scores...")

    scores = []

    for text in df["text"]:
        score = calculate_slop_score(text)
        scores.append(score)

    df["slop_score"] = scores

    # Current heuristic threshold
    # Score >= 20 = predicted slop
    df["predicted_slop"] = (
        df["slop_score"] >= 20
    ).astype(int)

    y_true = df["slop_label"]
    y_pred = df["predicted_slop"]

    print()
    print("=" * 70)
    print("SLOP SCORE EVALUATION")
    print("=" * 70)

    print()
    print("Average score by label:")

    print(
        df.groupby("slop_label")["slop_score"].mean()
    )

    print()
    print("Classification metrics:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Not Slop",
                "Slop",
            ],
            zero_division=0,
        )
    )

    print("Accuracy:")
    print(
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print()
    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print()
    print("=" * 70)
    print("MISCLASSIFIED EXAMPLES")
    print("=" * 70)

    errors = df[
        df["slop_label"] != df["predicted_slop"]
    ]

    print()
    print(
        f"Misclassified samples: {len(errors)}"
    )

    print()

    for _, row in errors.iterrows():

        actual = (
            "SLOP"
            if row["slop_label"] == 1
            else "NOT SLOP"
        )

        predicted = (
            "SLOP"
            if row["predicted_slop"] == 1
            else "NOT SLOP"
        )

        print("-" * 70)
        print(f"ID: {row['id']}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print(f"Score: {row['slop_score']}")
        print()
        print(row["text"][:500])

    print()
    print("=" * 70)


if __name__ == "__main__":
    evaluate_slop_score()