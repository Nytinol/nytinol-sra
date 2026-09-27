# `course_catalog.csv`

One row per course offered.

**Rows:** 72  ·  **Columns:** 12

## How it joins

`course_id` joins to `transcripts.csv`. `prerequisite_ids` refers to other `course_id` values in this same file.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `course_id` | string | Unique course identifier. Joins to transcripts.csv. |
| `subject` | string | Subject prefix, e.g. CMSC, IS, MATH. |
| `catalog_number` | string | Catalog number within the subject. |
| `course_title` | string | Course title. |
| `credits` | int | Credit hours the course carries. |
| `course_level` | enum | Lower division (100-200) or upper division (300-400). Values: `Lower`, `Upper`. |
| `course_type` | enum | Role the course plays in a degree. Values: `Core`, `Elective`, `General Education`, `Capstone`. |
| `required_for_majors` | list | Pipe-delimited majors that require this course. 'Not Applicable' if required by neither. |
| `prerequisite_ids` | list | Pipe-delimited course_ids that must be completed first. 'Not Applicable' if none. |
| `skill_tags` | list | Pipe-delimited skills the course covers. |
| `difficulty_index` | float | Relative difficulty, 1.0 gentle to 5.0 demanding. |
| `typical_terms_offered` | list | Pipe-delimited seasons the course is normally offered. |

## Sample rows

| `course_id` | `subject` | `catalog_number` | `course_title` | `credits` | `course_level` | `course_type` | `required_for_majors` | `prerequisite_ids` | `skill_tags` | `difficulty_index` | `typical_terms_offered` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMSC201 | CMSC | 201 | Foundations of Computer Science I | 4 | Lower | Core | Computer Science|Information Systems | Not Applicable | Python|Problem Solving | 2.6 | Fall|Spring|Summer |
| CMSC202 | CMSC | 202 | Foundations of Computer Science II | 4 | Lower | Core | Computer Science | CMSC201 | C++|Object-Oriented Design | 3.4 | Fall|Spring|Summer |
| CMSC203 | CMSC | 203 | Discrete Structures | 3 | Lower | Core | Computer Science | CMSC201 | Discrete Math|Proof Technique | 3.3 | Fall|Spring |

## Gotchas

- `prerequisite_ids`, `skill_tags`, `required_for_majors`, and `typical_terms_offered` are pipe-delimited (`|`). Split before analyzing.
- A prerequisite element may express alternatives with ` or `, as in `MATH151 or MATH155`. Either course satisfies the requirement.
- Courses with no prerequisites carry `Not Applicable`, not a blank.
- `difficulty_index` is a property of the course in this dataset, not a published UMBC figure.

## Five questions this file alone can answer

- Which courses have the most prerequisites, directly or transitively?
- Which skill_tags appear across the most courses?
- Which skills does the curriculum cover that roles rarely ask for, and vice versa?
- How does difficulty_index vary by course_level and subject?
- Which courses are only offered in one term per year?
- What does the prerequisite graph look like for each major's core?
