"""Analysis-ready alumni table.

No rows are dropped. The previous version kept only graduates with a full
employment record, which removed 1,168 of 3,200 people -- every graduate who went
to grad school, entered the military, was still looking, or did not respond. Those
are the outcomes a recommender has to be able to warn about, so they stay, and the
distinction between "we know they were not employed" and "we do not know" is kept
rather than collapsed.
"""

from __future__ import annotations

import paths

from model import data


def add_outcome_flags(alumni):
    """Separate a known non-employment outcome from an unknown one."""
    frame = alumni.copy()
    employed = frame["first_destination"].isin(
        ["Employed Full-Time", "Employed Part-Time"]
    )
    frame["is_employed_outcome"] = employed
    frame["outcome_is_known"] = frame["first_destination"] != "No Response"
    return frame


def add_real_salary(alumni):
    """First-job salary in 2026 dollars, so cohorts can be compared."""
    frame = alumni.copy()
    salary = frame["first_job_annual_salary_usd"]
    frame["first_job_salary_real_2026"] = [
        None if value != value else round(data.to_real_dollars(value, int(year)))
        for value, year in zip(salary, frame["graduation_year"])
    ]
    return frame


def main() -> None:
    alumni = data.load("alumni")
    alumni = add_outcome_flags(alumni)
    alumni = add_real_salary(alumni)
    paths.write(alumni, "alumni_clean")

    print("\nfirst_destination is kept in full:")
    print(alumni["first_destination"].value_counts().to_string())


if __name__ == "__main__":
    main()
