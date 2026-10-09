import pandas as pd


INPUT_PATH = "data/processed/hc3_prepared.csv"
OUTPUT_PATH = "data/processed/hc3_clean.csv"


MIN_LENGTH = 50
MAX_LENGTH = 10000


def clean_dataset():
    print("Loading prepared dataset...")

    df = pd.read_csv(INPUT_PATH)

    original_count = len(df)

    # Remove missing text
    df = df.dropna(subset=["text"])

    # Normalize whitespace
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove very short texts
    df = df[df["text"].str.len() >= MIN_LENGTH]

    # Remove extremely long texts
    df = df[df["text"].str.len() <= MAX_LENGTH]

    # Remove duplicate text
    df = df.drop_duplicates(subset=["text"])

    # Reset index
    df = df.reset_index(drop=True)

    print()
    print("Cleaning complete!")
    print(f"Original samples: {original_count}")
    print(f"Clean samples: {len(df)}")
    print(f"Removed samples: {original_count - len(df)}")

    print()
    print("AI label distribution:")
    print(df["ai_label"].value_counts())

    print()
    print("Text length statistics:")
    print(df["text"].str.len().describe())

    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_dataset()