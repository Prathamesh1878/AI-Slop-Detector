import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from config import DATA_COLUMNS


DATA_PATH = "data/raw/sample_dataset.csv"


def validate_dataset(df):
    print("Checking dataset...")

    missing_columns = [
        column for column in DATA_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        return False

    missing_text = df["text"].isna().sum()

    if missing_text > 0:
        print(f"Found {missing_text} rows with missing text.")
        return False

    invalid_ai = ~df["ai_label"].isin([0, 1])
    invalid_slop = ~df["slop_label"].isin([0, 1])

    if invalid_ai.any():
        print("Invalid AI labels found.")
        return False

    if invalid_slop.any():
        print("Invalid slop labels found.")
        return False

    print("Dataset validation passed!")
    return True


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)