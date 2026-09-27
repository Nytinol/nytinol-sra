"""Analysis-ready employment table.

Salaries arrive nominal for the year each spell began, so a 2015 figure and a 2026
figure are not the same unit. Two corrections are added alongside the original
column: one to 2026 dollars, and one further divided by the region's cost of living,
which is what makes a Bay Area offer and a Baltimore offer comparable.
"""

from __future__ import annotations

import paths

from model import data, terms


def add_term_index(employment):
    """Place each spell in the academic term it began, to align with transcripts."""
    frame = employment.copy()
    frame["start_term_index"] = frame["start_date"].map(terms.index_from_date)
    frame["start_year"] = frame["start_date"].str.slice(0, 4).astype(int)
    return frame


def add_real_salary(employment):
    frame = employment.copy()
    frame["salary_real_2026"] = [
        round(data.to_real_dollars(salary, year))
        for salary, year in zip(frame["annual_salary_usd"], frame["start_year"])
    ]
    frame["salary_real_col_adjusted"] = (
        frame["salary_real_2026"] / (frame["cost_of_living_index"] / 100)
    ).round()
    return frame


def add_spell_order(employment):
    """Which spell this is in the person's career, and whether it is the first."""
    frame = employment.sort_values(["campus_id", "start_date"]).copy()
    frame["spell_number"] = frame.groupby("campus_id").cumcount() + 1
    frame["is_first_job"] = frame["spell_number"] == 1
    return frame


def summarize(employment) -> None:
    print("\nnominal vs 2026 dollars, median first-job salary by start year:")
    first = employment[employment["is_first_job"]]
    table = first.groupby("start_year")[
        ["annual_salary_usd", "salary_real_2026"]
    ].median()
    print(table.round().to_string())

    print("\nchange_type counts, all spells kept:")
    print(employment["change_type"].value_counts().to_string())


def main() -> None:
    employment = data.load("employment_history")
    employment = add_term_index(employment)
    employment = add_real_salary(employment)
    employment = add_spell_order(employment)
    paths.write(employment, "employment_history_clean")
    summarize(employment)


if __name__ == "__main__":
    main()
