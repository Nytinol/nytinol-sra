import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "hackumbc-2026-main" / "data"

student_experience = pd.read_csv(DATA_DIR / "student_experience.csv")


def dropIsPaidCol(student_experience: pd.DataFrame) -> pd.DataFrame:
    student_experience = student_experience.drop(columns="is_paid", errors="ignore")
    return student_experience

def dropHoursPerWeekCol(student_experience: pd.DataFrame) -> pd.DataFrame:
    student_experience = student_experience.drop(columns="hours_per_week", errors="ignore")
    return student_experience 

def dropRoleNA(student_exp: pd.DataFrame) -> pd.DataFrame:
    keep = student_exp["role_level"].ne("Not Applicable")
    df = student_exp.loc[keep]
    return df

student_experience = dropIsPaidCol(student_experience)
student_experience = dropHoursPerWeekCol(student_experience)
student_experience = dropRoleNA(student_experience)
student_experience.to_csv(DATA_DIR / "student_experience.csv", index=False)