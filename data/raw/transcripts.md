# `transcripts.csv`

One row per course attempt, for both current students and alumni.

**Rows:** 140,458  ·  **Columns:** 11

## How it joins

`campus_id` joins to exactly one of `students_current.csv` or `alumni.csv`. `course_id` joins to `course_catalog.csv`.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `campus_id` | string | Person this attempt belongs to. |
| `term` | term | Term the course was attempted. |
| `course_id` | string | Joins to course_catalog.csv. |
| `course_title` | string | Denormalized from the catalog so this file is useful without a join. |
| `subject` | string | Subject prefix, denormalized from the catalog. |
| `credits_attempted` | int | Credits the course carries. |
| `credits_earned` | int | Credits awarded: equal to credits_attempted for grades A-D, 0 for F, W, and IP. |
| `grade` | enum | Letter grade on a straight 4.0 scale with no plus/minus. W is a withdrawal, IP an in-progress Fall 2026 course. Values: `A`, `B`, `C`, `D`, `F`, `W`, `IP`. |
| `grade_points` | float | Quality points per credit: A=4, B=3, C=2, D=1, F=0. 'Not Applicable' for W and IP, which are excluded from GPA. |
| `is_repeat` | bool | TRUE if this person attempted this course in an earlier term. Values: `TRUE`, `FALSE`. |
| `requirement_category` | enum | How this course counts toward the degree. Values: `Major Core`, `Major Elective`, `Supporting Coursework`, `General Education`, `Free Elective`. |

## Sample rows

| `campus_id` | `term` | `course_id` | `course_title` | `subject` | `credits_attempted` | `credits_earned` | `grade` | `grade_points` | `is_repeat` | `requirement_category` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CID-116490 | Fall 2025 | MATH151 | Calculus and Analytic Geometry I | MATH | 4 | 4 | B | 3.0 | FALSE | Supporting Coursework |
| CID-116490 | Fall 2025 | CMSC202 | Foundations of Computer Science II | CMSC | 4 | 4 | B | 3.0 | FALSE | Major Core |
| CID-116490 | Fall 2025 | CMSC203 | Discrete Structures | CMSC | 3 | 3 | A | 4.0 | FALSE | Major Core |

## Gotchas

- `W` (withdrawal) and `IP` (in progress) rows carry `grade_points` of `Not Applicable` and are excluded from GPA. Filter them out before averaging, or the column will not parse as numeric.
- `IP` rows exist only for current students in Fall 2026.
- A repeated course appears once per attempt. `is_repeat` marks the second and later attempts, so counting distinct `course_id` per person is not the same as counting rows.
- Transferred credits are not represented here. A `Transfer` student's transcript begins at UMBC.

## Five questions this file alone can answer

- Which courses have the lowest average grade points?
- Which courses are most often repeated?
- How does average grade vary by requirement_category?
- Which subjects carry the most credits across the whole dataset?
- How does withdrawal rate (grade = W) vary by course and by term?
