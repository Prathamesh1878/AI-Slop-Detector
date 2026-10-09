import pandas as pd

from slop_features import calculate_slop_features


INPUT_PATH = "../../data/processed/train.csv"
OUTPUT_PATH = "../../data/processed/train_slop_features.csv"


def analyze_features():
    print("Loading training dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Samples: {len(df)}")
    print()
    print("Calculating slop features...")

    feature_rows = []

    for text in df["text"]:
        feature_rows.append(calculate_slop_features(text))

    feature_df = pd.DataFrame(feature_rows)

    result = pd.concat(
        [
            df[["text", "ai_label"]],
            feature_df,
        ],
        axis=1,
    )

    print()
    print("Feature extraction complete!")

    print()
    print("Average features by label:")
    print(
        result.groupby("ai_label")[
            [
                "word_count",
                "sentence_count",
                "avg_sentence_length",
                "buzzword_count",
                "buzzword_density",
                "filler_phrase_count",
                "repetition_ratio",
                "question_ratio",
                "exclamation_ratio",
            ]
        ].mean()
    )

    print()
    print("Saving feature dataset...")

    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    analyze_features()