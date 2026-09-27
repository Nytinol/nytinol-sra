"""Loading and summarizing all six files with pandas.

    python3 examples/explore_python.py

One of several ways in -- see explore_sql.sql for the same ideas in SQL, and
quickstart.py for a version with no third-party dependencies at all. Nothing
about this dataset requires Python or pandas.

Prints results rather than plotting them. What you visualize, and whether you
visualize at all, is yours to decide.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def load_all():
    """All six files. Roughly a second and ~200 MB of RAM."""
    return {
        "students": pd.read_csv(os.path.join(DATA, "students_current.csv")),
        "alumni": pd.read_csv(os.path.join(DATA, "alumni.csv")),
        "transcripts": pd.read_csv(os.path.join(DATA, "transcripts.csv")),
        "jobs": pd.read_csv(os.path.join(DATA, "employment_history.csv")),
        "experience": pd.read_csv(os.path.join(DATA, "student_experience.csv")),
        "catalog": pd.read_csv(os.path.join(DATA, "course_catalog.csv")),
    }


def employed_alumni(alumni):
    """Alumni with a first job, salary cast to an integer.

    The raw column is text because non-employed graduates carry the literal
    string "Not Applicable".
    """
    employed = alumni[alumni["first_job_annual_salary_usd"] != "Not Applicable"].copy()
    employed["salary"] = employed["first_job_annual_salary_usd"].astype(int)
    return employed


def graded(transcripts):
    """Transcript rows that count toward GPA: W and IP excluded."""
    rows = transcripts[~transcripts["grade"].isin(["W", "IP"])].copy()
    rows["grade_points"] = rows["grade_points"].astype(float)
    return rows


def rule(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    data = load_all()
    print("Loaded:")
    for name, frame in data.items():
        print("  %-12s %8s rows x %2d cols"
              % (name, format(len(frame), ","), frame.shape[1]))

    catalog, transcripts = data["catalog"], data["transcripts"]
    jobs, experience = data["jobs"], data["experience"]

    # ---- the curriculum, from one file -----------------------------------
    rule("Courses that gate the most other courses")
    prereqs = catalog[catalog["prerequisite_ids"] != "Not Applicable"]
    gates = prereqs["prerequisite_ids"].str.split("|").explode().value_counts()
    titles = catalog.set_index("course_id")["course_title"]
    for course_id, count in gates.head(8).items():
        print("  %-9s unlocks %2d  %s" % (course_id, count,
                                          titles.get(course_id, "")[:42]))

    # ---- grades, from one file -------------------------------------------
    rule("Lowest average grade points (200+ attempts)")
    stats = graded(transcripts).groupby(["course_id", "course_title"]).agg(
        avg=("grade_points", "mean"), n=("grade_points", "size"))
    for (course_id, title), row in stats[stats["n"] >= 200] \
            .sort_values("avg").head(8).iterrows():
        print("  %.2f  %-9s %-44s n=%d" % (row["avg"], course_id, title[:44],
                                           row["n"]))

    # ---- involvement, from one file --------------------------------------
    rule("What students do outside class")
    for kind, count in experience["experience_type"].value_counts().items():
        print("  %-24s %6s" % (kind, format(count, ",")))

    # ---- careers, from one file ------------------------------------------
    rule("Most common role-to-role transitions")
    ordered = jobs.sort_values(["campus_id", "start_date"])
    ordered["next_role"] = ordered.groupby("campus_id")["job_title"].shift(-1)
    edges = (ordered.dropna(subset=["next_role"])
                    .groupby(["job_title", "next_role"]).size()
                    .sort_values(ascending=False))
    for (source, target), weight in edges.head(8).items():
        print("  %3d  %-33s -> %s" % (weight, source[:33], target[:33]))

    # ---- what roles ask for, versus what courses teach -------------------
    rule("Skills roles ask for that no course covers")
    course_skills = set(catalog["skill_tags"].str.split("|").explode())
    role_skills = jobs["role_skill_tags"].str.split("|").explode().value_counts()
    uncovered = [s for s in role_skills.index if s not in course_skills]
    print("  %s" % (", ".join(uncovered) if uncovered
                    else "none -- every skill roles ask for is taught somewhere"))

    rule("Skills taught that roles rarely ask for")
    asked = set(role_skills.index)
    unasked = sorted(s for s in course_skills if s and s not in asked)
    print("  %d of %d course skills: %s%s"
          % (len(unasked), len(course_skills), ", ".join(unasked[:12]),
             " ..." if len(unasked) > 12 else ""))

    print("\nThose last two are a starting point, not a finding. Whether an "
          "\"uncovered\" skill\nmatters depends on questions this data cannot "
          "answer on its own.")


if __name__ == "__main__":
    main()
