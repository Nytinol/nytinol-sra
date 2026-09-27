import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "hackumbc-2026-main" / "data"

employment_history = pd.read_csv(DATA_DIR / "employment_history.csv")


def print_unique_values(employment_history: pd.DataFrame) -> None:
    for column in employment_history.columns:
        counts = employment_history[column].value_counts(dropna=False)
        print(f"{column} ({counts.size} unique)")
        print(counts.head(10))
        print()




