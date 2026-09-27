# `students_current.csv`

One row per student enrolled in Fall 2026.

**Rows:** 1,800  ·  **Columns:** 22

## How it joins

`campus_id` joins to `transcripts.csv` and `student_experience.csv`. It never appears in `alumni.csv` or `employment_history.csv`.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `campus_id` | string | Unique person identifier, format CID-NNNNNN. Joins to every other file. A campus_id appears in either students_current.csv or alumni.csv, never both. |
| `entry_term` | term | Term the student first enrolled at UMBC, e.g. 'Fall 2023'. |
| `entry_type` | enum | Whether the student started here or transferred in. Values: `First-Time Freshman`, `Transfer`. |
| `major` | enum | Primary major. Values: `Computer Science`, `Information Systems`. |
| `track` | string | Concentration within the major. 'General' where no track is declared. |
| `second_major` | string | Second major, or 'Not Applicable'. |
| `minor` | string | Declared minor, or 'Not Applicable'. |
| `class_level` | enum | Standing by credits earned: Freshman <30, Sophomore 30-59, Junior 60-89, Senior 90+. Values: `Freshman`, `Sophomore`, `Junior`, `Senior`. |
| `residency` | enum | Tuition residency status. Values: `In-State`, `Out-of-State`. |
| `enrollment_intensity` | enum | Full-Time is 12 or more credits in the current term. Values: `Full-Time`, `Part-Time`. |
| `is_first_generation` | bool | TRUE if neither parent completed a bachelor's degree. Values: `TRUE`, `FALSE`. |
| `work_hours_per_week` | int | Hours of paid employment per week during the term, 0-40. |
| `credits_earned` | int | Credits earned toward the degree as of the start of Fall 2026. |
| `credits_required` | int | Credits required for the degree, typically 120. |
| `cumulative_gpa` | float | GPA across all graded attempts, 0.00-4.00, two decimals. 'Not Applicable' for students in their first term, who have not completed any graded coursework yet. |
| `major_gpa` | float | GPA across courses in the major subject only. 'Not Applicable' for students in their first term. |
| `academic_standing` | enum | Standing as of the current term. Values: `Good Standing`, `Academic Warning`, `Academic Probation`. |
| `expected_graduation_term` | term | Projected graduation term given current pace. |
| `internship_count` | int | Internship and co-op records for this student in student_experience.csv. |
| `credential_count` | int | Certification records for this student in student_experience.csv. |
| `engagement_activity_count` | int | All other experience records: clubs, research, hackathons, tutoring, campus jobs, peer mentoring, competitive teams. |
| `tuition_paid_to_date_usd` | int | Cumulative net tuition and fees paid so far, after aid. |

## Sample rows

| `campus_id` | `entry_term` | `entry_type` | `major` | `track` | `second_major` | `minor` | `class_level` | `residency` | `enrollment_intensity` | `is_first_generation` | `work_hours_per_week` | `credits_earned` | `credits_required` | `cumulative_gpa` | `major_gpa` | `academic_standing` | `expected_graduation_term` | `internship_count` | `credential_count` | `engagement_activity_count` | `tuition_paid_to_date_usd` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CID-116490 | Fall 2025 | Transfer | Computer Science | Data Science | Not Applicable | Not Applicable | Junior | In-State | Full-Time | TRUE | 20 | 76 | 120 | 3.00 | 3.00 | Good Standing | Spring 2028 | 1 | 2 | 4 | 8550 |
| CID-227285 | Fall 2026 | First-Time Freshman | Computer Science | Artificial Intelligence | Not Applicable | Not Applicable | Freshman | In-State | Full-Time | FALSE | 21 | 0 | 120 | Not Applicable | Not Applicable | Good Standing | Spring 2031 | 0 | 0 | 3 | 0 |
| CID-514009 | Fall 2025 | First-Time Freshman | Information Systems | Cybersecurity Management | Not Applicable | Not Applicable | Freshman | In-State | Full-Time | FALSE | 9 | 29 | 120 | 2.66 | 4.00 | Good Standing | Spring 2030 | 1 | 0 | 5 | 13500 |

## Gotchas

- `cumulative_gpa` excludes in-progress (`IP`) Fall 2026 coursework, so it will not match a naive average over every transcript row for this person.
- `credits_earned` includes credits transferred in from another institution for students whose `entry_type` is `Transfer`. Those transferred credits do **not** appear as transcript rows, so summing `transcripts.csv` alone will under-count them.
- The three `*_count` columns duplicate information in `student_experience.csv` on purpose, so this file is useful on its own.

## Five questions this file alone can answer

- What is the distribution of cumulative GPA across majors?
- How many credits separate the median junior from the median senior?
- How does work_hours_per_week vary by residency and first-generation status?
- Which tracks have the most students, and how does that differ by class level?
- What share of students are on pace to graduate before their expected term?
