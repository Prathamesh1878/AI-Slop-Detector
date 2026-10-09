from datasets import load_dataset


DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Hello-SimpleAI/HC3/resolve/"
    "refs%2Fconvert%2Fparquet/"
    "all/train/0000.parquet"
)


def download_hc3():
    print("Downloading HC3 dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": DATASET_URL},
        split="train",
    )

    print()
    print("Dataset downloaded successfully!")
    print()
    print(dataset)

    print()
    print("Columns:")
    print(dataset.column_names)

    print()
    print("Number of rows:")
    print(len(dataset))

    print()
    print("First row:")
    print(dataset[0])

    return dataset


if __name__ == "__main__":
    download_hc3()