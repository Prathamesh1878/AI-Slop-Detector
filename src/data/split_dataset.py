import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/hc3_clean.csv"

TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

RANDOM_STATE = 42


def split_dataset():
    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Total samples: {len(df)}")

    # First split: 80% train, 20% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["ai_label"],
        random_state=RANDOM_STATE,
    )

    # Second split: temporary 20% into 10% validation and 10% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["ai_label"],
        random_state=RANDOM_STATE,
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print()
    print("Dataset split complete!")
    print()

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples:       {len(test_df)}")

    print()
    print("Training label distribution:")
    print(train_df["ai_label"].value_counts())

    print()
    print("Validation label distribution:")
    print(validation_df["ai_label"].value_counts())

    print()
    print("Test label distribution:")
    print(test_df["ai_label"].value_counts())

    print()
    print("Saved:")
    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    split_dataset()