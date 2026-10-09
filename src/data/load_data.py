import pandas as pd


DATA_PATH = "data/raw/sample_dataset.csv"


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


if __name__ == "__main__":
    df = load_dataset()

    print("Dataset loaded successfully!")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print()
    print(df.head())