# The data

Six CSV files. Each has a `.md` beside it with every field, three sample rows, and the gotchas specific to that file.

| File | Rows | One row is | Docs |
| --- | --- | --- | --- |
| `students_current.csv` | 1,800 | a student enrolled in Fall 2026 | [students_current.md](students_current.md) |
| `alumni.csv` | 3,200 | a graduate, highest degree only | [alumni.md](alumni.md) |
| `transcripts.csv` | ~138,000 | one course attempt | [transcripts.md](transcripts.md) |
| `employment_history.csv` | ~6,000 | one job spell | [employment_history.md](employment_history.md) |
| `student_experience.csv` | ~20,000 | one activity | [student_experience.md](student_experience.md) |
| `course_catalog.csv` | 72 | one course | [course_catalog.md](course_catalog.md) |

`sample/` holds a referentially complete 10% cut with identical filenames and columns.

---

## The join graph

```
                       course_catalog.csv
                              │ course_id
                              ▼
  students_current.csv ──┐  transcripts.csv
        (1,800)          │    (~138,000)
                         ├───────┤ campus_id
        alumni.csv ──────┘       │
        (3,200)          ├───────┤
                         │       │
   employment_history.csv│  student_experience.csv
        (~6,000)         │      (~20,000)
     alumni only         │   students and alumni
```

`campus_id` (format `CID-123456`) is the only person key. It appears in `students_current.csv` **or** `alumni.csv`, never both.

`course_id` joins `transcripts.csv` to `course_catalog.csv`.

**A connection that isn't a join:** `skill_tags` in `course_catalog.csv` and `role_skill_tags` in `employment_history.csv` draw from the same vocabulary of 119 skills. What a course teaches and what a role asks for are directly comparable — set operations on the split lists, no mapping table needed.

## Which file covers whom

| | students_current | alumni | transcripts | employment_history | student_experience |
| --- | --- | --- | --- | --- | --- |
| **Current students** | ✓ | | ✓ | | ✓ |
| **Alumni** | | ✓ | ✓ | ✓ | ✓ |

Not every alum appears in `employment_history.csv` — those who went to graduate school, entered the military, were still seeking, or did not respond have no job rows.

---

## Conventions used everywhere

**`Not Applicable` means "does not apply."** It is a literal string. A certification has no `hours_per_week`; an alum who went to graduate school has no `first_job_title`. Those cells read `Not Applicable`, and the columns containing them load as text — so filter before casting.

The phrase is spelled out rather than abbreviated on purpose: pandas converts a literal `N/A` to `NaN` by default, which would turn every such filter into a silent no-op.

**The only true blank is `end_date`** in `employment_history.csv`, and it is blank if and only if `is_current` is true. It is the only `NaN` in the dataset.

**Booleans are `TRUE` / `FALSE`.** pandas reads these as real booleans, so compare with `== True` rather than `== "TRUE"`. Columns that also carry `Not Applicable` stay as text.

**Terms look like `Fall 2023`.** Seasons are `Spring`, `Summer`, `Fall` in that order within a year. Most enrollment is Fall and Spring.

**Dates are `YYYY-MM-DD`.**

**Pipe-delimited lists** appear in `course_catalog.csv` (`skill_tags`, `prerequisite_ids`, `required_for_majors`, `typical_terms_offered`) and in `employment_history.csv` (`role_skill_tags`): `Python|SQL|Data Modeling`.

**Money is whole dollars, no symbols or separators.** Salaries are nominal for the year the job started.

**Grades are `A B C D F W IP`** on a straight 4.0 scale — no plus/minus. `W` is a withdrawal and `IP` is an in-progress Fall 2026 course; both carry `Not Applicable` grade points and are excluded from GPA.

**The dataset's "today" is September 15, 2026.** Tenure of current jobs is measured to that date.

**Students in their first term have no GPA.** About 400 students entered in Fall 2026, so all of their coursework is still `IP` and their `cumulative_gpa` and `major_gpa` read `Not Applicable` rather than `0.00`.

---

## First fifteen minutes

These are plain CSV files. Load them however you like — the examples below happen to be Python and SQL, but nothing here depends on either.

```python
import pandas as pd
alumni = pd.read_csv("alumni.csv")

# The salary column is text because non-employed graduates carry a sentinel.
employed = alumni[alumni["first_job_annual_salary_usd"] != "Not Applicable"].copy()
employed["salary"] = employed["first_job_annual_salary_usd"].astype(int)
employed.groupby("graduation_year")["salary"].median()
```

```sql
SELECT graduation_year, COUNT(*) AS graduates
FROM alumni GROUP BY graduation_year ORDER BY graduation_year;
```

Either is a real result from one file with no joins. Every one of the six files can do something like it — see the "Five questions this file alone can answer" section in each `.md`.

---

## Recomputing derived columns

Several columns are derived, and you can check them against their sources. They agree by construction.

| Column | Recompute from |
| --- | --- |
| `cumulative_gpa` | `transcripts.csv`, credit-weighted over rows where `grade` is not `W` or `IP` |
| `credits_earned` | sum of `credits_earned` in `transcripts.csv` (**plus** transferred credits for `Transfer` entrants, which have no transcript rows) |
| `internship_count` | rows in `student_experience.csv` where `experience_type` is `Internship` or `Co-op` |
| `credential_count` | rows where `experience_type` is `Certification` |
| `engagement_activity_count` | all other `student_experience.csv` rows |
| `tenure_months` | `start_date` to `end_date`, or to 2026-09-15 when `is_current` |

The three `*_count` columns are duplicated onto the person files deliberately, so you can relate involvement to outcomes without a join.
