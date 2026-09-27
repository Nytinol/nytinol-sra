import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "hackumbc-2026-main" / "data"

transcripts = pd.read_csv(DATA_DIR / "transcripts.csv")


def dropGradePointsCol(transcripts: pd.DataFrame) -> pd.DataFrame:
    transcripts = transcripts.drop(columns="grade_points", errors="ignore")
    return transcripts

def dropCreditsEarnedCol(transcripts: pd.DataFrame) -> pd.DataFrame:
    transcripts = transcripts.drop(columns="credits_earned", errors="ignore")
    return transcripts


transcripts = dropGradePointsCol(transcripts)
transcripts = dropCreditsEarnedCol(transcripts)
transcripts.to_csv(DATA_DIR / "transcripts.csv", index=False)

