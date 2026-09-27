import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "hackumbc-2026-main" / "data"

alumni = pd.read_csv(DATA_DIR / "alumni.csv")


SENTINEL_COLUMNS = [
    "months_to_first_job",
    "first_job_title",
    "first_job_family",
    "first_employer",
    "first_employer_industry",
    "first_job_region",
    "first_job_annual_salary_usd",
    "first_job_is_remote",
    "first_job_found_via",
]


def dropNA(alumni: pd.DataFrame) -> pd.DataFrame:
    keep = (alumni["first_destination"] != "No Response") & alumni[SENTINEL_COLUMNS].ne(
        "Not Applicable"
    ).all(axis=1)
    df = alumni.loc[keep].dropna(subset=["first_destination", *SENTINEL_COLUMNS])

    for column in df.columns:
        print(f"{column} Not Applicable: {(df[column] == 'Not Applicable').sum()}")
        print(f"{column} No Response: {(df[column] == 'No Response').sum()}")
        print()

    return df

def removeMasters(alumni: pd.DataFrame):
    keep = alumni["degree_level"].ne("Master of Science")
    df = alumni.loc[keep]
    print(f"{df["degree_level"]} Not Applicable: {(df["degree_level"] == 'Master of Science').sum()}")
    print(f"{df["degree_level"]} No Response: {(df["degree_level"] == 'Bachelor of Science').sum()}")
    print()

    return df




df = dropNA(alumni)
df = removeMasters(df)
df.to_csv(DATA_DIR / "alumni.csv", index=False)
