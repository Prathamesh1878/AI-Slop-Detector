import pandas as pd


INPUT_PATH = "data/processed/hc3_clean.csv"
OUTPUT_PATH = "data/raw/slop_annotation_set.csv"

SAMPLE_SIZE_PER_LABEL = 50


def create_annotation_set():
    df = pd.read_csv(INPUT_PATH)

    human = (
        df[df["ai_label"] == 0]
        .sample(SAMPLE_SIZE_PER_LABEL, random_state=42)
    )

    ai = (
        df[df["ai_label"] == 1]
        .sample(SAMPLE_SIZE_PER_LABEL, random_state=42)
    )

    samples = pd.concat([human, ai])
    samples = samples.sample(frac=1, random_state=42)
    samples = samples.reset_index(drop=True)

    annotation_df = pd.DataFrame({
        "id": range(len(samples)),
        "text": samples["text"],
        "ai_label": samples["ai_label"],
        "slop_label": "",
        "slop_reason": "",
    })

    annotation_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Slop annotation set created!")
    print()
    print(f"Total samples: {len(annotation_df)}")
    print()
    print("AI distribution:")
    print(annotation_df["ai_label"].value_counts())
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_annotation_set() 