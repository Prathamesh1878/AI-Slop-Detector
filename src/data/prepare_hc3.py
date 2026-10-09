from datasets import load_dataset
import pandas as pd


DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Hello-SimpleAI/HC3/resolve/"
    "refs%2Fconvert%2Fparquet/"
    "all/train/0000.parquet"
)


def prepare_hc3():
    print("Loading HC3 dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": DATASET_URL},
        split="train",
    )

    rows = []

    for item in dataset:
        source = item["source"]

        for text in item["human_answers"]:
            if text and text.strip():
                rows.append({
                    "text": text.strip(),
                    "ai_label": 0,
                    "source": source,
                    "domain": "general",
                })

        for text in item["chatgpt_answers"]:
            if text and text.strip():
                rows.append({
                    "text": text.strip(),
                    "ai_label": 1,
                    "source": source,
                    "domain": "general",
                })

    df = pd.DataFrame(rows)

    print()
    print("HC3 preparation complete!")
    print(f"Total text samples: {len(df)}")
    print()
    print("AI label distribution:")
    print(df["ai_label"].value_counts())
    print()
    print("First 5 samples:")
    print(df.head())

    output_path = "data/processed/hc3_prepared.csv"

    df.to_csv(output_path, index=False)

    print()
    print(f"Saved prepared dataset to: {output_path}")


if __name__ == "__main__":
    prepare_hc3()