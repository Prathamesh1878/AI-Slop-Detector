import pandas as pd
from pathlib import Path

INPUT_PATH = "data/processed/hc3_clean.csv"
EXISTING_PATH = "data/raw/slop_annotation_set.csv"
OUTPUT_PATH = "data/raw/slop_annotation_batch2.csv"

SAMPLES_PER_CLASS = 200
RANDOM_STATE = 42


def create_annotation_batch():
    print("Loading cleaned HC3 dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Loading existing annotations...")
    existing = pd.read_csv(EXISTING_PATH)

    existing_texts = set(existing["text"].astype(str))

    # Remove anything already manually annotated
    df = df[~df["text"].astype(str).isin(existing_texts)].copy()

    print(f"Available after removing existing examples: {len(df)}")

    human = df[df["ai_label"] == 0]
    ai = df[df["ai_label"] == 1]

    human_sample = human.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    ai_sample = ai.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    batch = pd.concat(
        [human_sample, ai_sample],
        ignore_index=True,
    )

    # Add IDs for annotation
    batch.insert(
        0,
        "id",
        range(
            len(existing),
            len(existing) + len(batch),
        ),
    )

    # Empty column for manual slop labeling
    batch["slop_label"] = ""

    # Keep a predictable column order
    batch = batch[
        [
            "id",
            "text",
            "ai_label",
            "slop_label",
            "source",
            "domain",
        ]
    ]

    # Shuffle human/AI examples
    batch = batch.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Annotation batch created!")
    print(f"Total examples: {len(batch)}")
    print(f"Human examples: {(batch['ai_label'] == 0).sum()}")
    print(f"AI examples: {(batch['ai_label'] == 1).sum()}")
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_annotation_batch()