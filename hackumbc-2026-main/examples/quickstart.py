"""Zero-install quickstart -- Python standard library only.

    python3 examples/quickstart.py

Prints four summaries using nothing but `csv`. No pandas, no `pip install`.
If this runs, your environment can handle the whole dataset.

One of three examples in this folder, none of which is the required
approach: explore_python.py does the same work in pandas, explore_sql.sql
does it in SQL, and the files are plain CSV besides -- use whatever you
already know.
"""

import csv
import os
from collections import Counter, defaultdict

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load(name):
    path = os.path.join(DATA, name)
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def money(value):
    return "$%s" % format(int(value), ",")


def median(values):
    values = sorted(values)
    if not values:
        return 0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def rule(title):
    print("\n" + title)
    print("-" * len(title))


# --------------------------------------------------------------------------
# 1. alumni.csv -- median first-job salary by graduation year
# --------------------------------------------------------------------------

alumni = load("alumni.csv")
rule("Median first-job salary by graduation year")

by_year = defaultdict(list)
for row in alumni:
    # "Not Applicable" where the graduate was not employed -- filter before casting.
    if row["first_job_annual_salary_usd"] != "Not Applicable":
        by_year[int(row["graduation_year"])].append(
            int(row["first_job_annual_salary_usd"]))

for year in sorted(by_year):
    salaries = by_year[year]
    bar = "#" * int(median(salaries) / 2500)
    print("  %d  n=%4d  %9s  %s" % (year, len(salaries),
                                    money(median(salaries)), bar))


# --------------------------------------------------------------------------
# 2. transcripts.csv -- hardest courses by average grade points
# --------------------------------------------------------------------------

transcripts = load("transcripts.csv")
rule("Ten lowest average grade points (100+ attempts)")

points = defaultdict(list)
for row in transcripts:
    # W and IP carry no grade points and are excluded from GPA.
    if row["grade"] not in ("W", "IP"):
        points[(row["course_id"], row["course_title"])].append(
            float(row["grade_points"]))

ranked = [(sum(v) / len(v), len(v), k) for k, v in points.items() if len(v) >= 100]
for average, n, (course_id, title) in sorted(ranked)[:10]:
    print("  %.2f  %-9s %-46s n=%d" % (average, course_id, title[:46], n))


# --------------------------------------------------------------------------
# 3. employment_history.csv -- most common job transitions
# --------------------------------------------------------------------------

jobs = load("employment_history.csv")
rule("Most common role-to-role transitions")

by_person = defaultdict(list)
for row in jobs:
    by_person[row["campus_id"]].append(row)

transitions = Counter()
for spells in by_person.values():
    spells.sort(key=lambda r: r["start_date"])
    for current, following in zip(spells, spells[1:]):
        transitions[(current["job_title"], following["job_title"])] += 1

for (source, target), count in transitions.most_common(10):
    print("  %3d  %-34s -> %s" % (count, source[:34], target[:34]))


# --------------------------------------------------------------------------
# 4. student_experience.csv -- what students actually do
# --------------------------------------------------------------------------

experience = load("student_experience.csv")
rule("Experience records by type")

counts = Counter(row["experience_type"] for row in experience)
widest = max(len(k) for k in counts)
for kind, count in counts.most_common():
    print("  %-*s %6d  %s" % (widest, kind, count, "#" * (count // 250)))


print("\nLoaded %s alumni, %s transcript rows, %s job spells, %s activities."
      % (format(len(alumni), ","), format(len(transcripts), ","),
         format(len(jobs), ","), format(len(experience), ",")))
print("Next: open data/README.md, then the .md beside whichever file interests you.")
