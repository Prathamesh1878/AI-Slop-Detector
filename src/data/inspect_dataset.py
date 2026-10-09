import pandas as pd


DATA_PATH = "data/processed/hc3_prepared.csv"


def inspect_dataset():
    print("Loading prepared dataset...")

    df = pd.read_csv(DATA_PATH)

    print()
    print("Dataset shape:")
    print(df.shape)

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("Missing values:")
    print(df.isnull().sum())

    print()
    print("Duplicate rows:")
    print(df.duplicated().sum())

    print()
    print("AI label distribution:")
    print(df["ai_label"].value_counts())

    print()
    print("Text length statistics:")
    print(df["text"].str.len().describe())

    print()
    print("Shortest samples:")
    print(df.loc[df["text"].str.len().nsmallest(5).index, "text"])

    print()
    print("Longest sample length:")
    print(df["text"].str.len().max())


if __name__ == "__main__":
    inspect_dataset()