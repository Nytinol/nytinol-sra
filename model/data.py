"""Typed loaders over the pristine dataset.

`data/raw/` is read-only and is the single source of truth. Every table is
normalized here rather than in each caller, because the dataset spells missing
values as the literal string `Not Applicable`, which silently turns numeric
columns into text.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
DERIVED_DIR = ROOT_DIR / "data" / "derived"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

NOT_APPLICABLE = "Not Applicable"

NUMERIC_COLUMNS: dict[str, list[str]] = {
    "alumni": [
        "graduation_year",
        "time_to_degree_years",
        "total_credits_earned",
        "final_gpa",
        "major_gpa",
        "work_hours_per_week_while_enrolled",
        "internship_count",
        "credential_count",
        "engagement_activity_count",
        "net_cost_usd",
        "total_loans_usd",
        "months_to_first_job",
        "first_job_annual_salary_usd",
    ],
    "students_current": [
        "work_hours_per_week",
        "credits_earned",
        "credits_required",
        "cumulative_gpa",
        "major_gpa",
        "internship_count",
        "credential_count",
        "engagement_activity_count",
        "tuition_paid_to_date_usd",
    ],
    "transcripts": ["credits_attempted", "credits_earned", "grade_points"],
    "employment_history": ["cost_of_living_index", "tenure_months", "annual_salary_usd"],
    "student_experience": ["duration_terms", "hours_per_week"],
    "course_catalog": ["credits", "difficulty_index"],
}

BOOLEAN_COLUMNS: dict[str, list[str]] = {
    "alumni": ["holds_prior_umbc_bachelors", "first_job_is_remote"],
    "students_current": ["is_first_generation"],
    "transcripts": ["is_repeat"],
    "employment_history": ["is_remote", "is_current", "requires_clearance"],
    "student_experience": ["is_paid"],
}

PIPE_COLUMNS: dict[str, list[str]] = {
    "course_catalog": [
        "skill_tags",
        "prerequisite_ids",
        "required_for_majors",
        "typical_terms_offered",
    ],
    "employment_history": ["role_skill_tags"],
}

TABLES = tuple(
    [
        "alumni",
        "students_current",
        "transcripts",
        "employment_history",
        "student_experience",
        "course_catalog",
    ]
)


def _normalize(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC_COLUMNS.get(name, []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].replace(NOT_APPLICABLE, None), errors="coerce"
            )
    for column in BOOLEAN_COLUMNS.get(name, []):
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .replace({"TRUE": True, "FALSE": False, NOT_APPLICABLE: None})
                .astype("boolean")
            )
    return frame


@lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    """Load one table, typed. Cached, so callers may call this freely."""
    if name not in TABLES:
        raise KeyError(f"Unknown table {name!r}; expected one of {TABLES}")
    return _normalize(name, pd.read_csv(RAW_DIR / f"{name}.csv"))


def split_pipe(value: object) -> list[str]:
    """Split a pipe-delimited cell, treating the sentinel as empty."""
    if value is None or not isinstance(value, str) or value == NOT_APPLICABLE:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


@lru_cache(maxsize=1)
def wage_index() -> dict[int, float]:
    """Empirical entry-level wage index, 2026 = 1.0.

    Salaries are nominal for the year the job started, so comparing 2015 to
    2026 compares different dollars. Rather than import an external CPI series,
    this uses median first-job salary by graduation cohort: those are all
    entry-level by construction, so the index is not contaminated by the panel
    maturing into senior roles.
    """
    alumni = load("alumni")
    employed = alumni.dropna(subset=["first_job_annual_salary_usd"])
    medians = employed.groupby("graduation_year")["first_job_annual_salary_usd"].median()
    # Smooth the cohort noise; the underlying drift is monotone.
    smoothed = medians.rolling(3, center=True, min_periods=1).mean()
    base = float(smoothed.loc[smoothed.index.max()])
    return {int(year): float(value) / base for year, value in smoothed.items()}


def to_real_dollars(salary: float | np.ndarray, year: int | np.ndarray):
    """Convert nominal salary of `year` into 2026 dollars."""
    index = wage_index()
    years = sorted(index)
    lowest, highest = years[0], years[-1]

    def factor(single_year: int) -> float:
        clamped = min(max(int(single_year), lowest), highest)
        return index[clamped]

    if isinstance(year, np.ndarray):
        factors = np.array([factor(value) for value in year], dtype=float)
        return np.asarray(salary, dtype=float) / factors
    return float(salary) / factor(year)


def materialize_derived() -> None:
    """Write typed copies of every table to `data/derived/`.

    Nothing in the pipeline requires these files; they exist so the tables can
    be inspected, loaded into SQL, or uploaded without re-deriving the types.
    """
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        frame = load(name)
        frame.to_csv(DERIVED_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    materialize_derived()
