# `student_experience.csv`

One row per activity, covering both current students and alumni.

**Rows:** 20,059  ·  **Columns:** 12

## How it joins

`campus_id` joins to either `students_current.csv` or `alumni.csv`.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `record_id` | string | Unique activity identifier, format EXP-NNNNNNN. |
| `campus_id` | string | Person this activity belongs to. |
| `experience_type` | enum | Kind of activity. Values: `Internship`, `Co-op`, `Undergraduate Research`, `Student Organization`, `Hackathon`, `Tutoring`, `Campus Job`, `Peer Mentor`, `Competitive Team`, `Certification`. |
| `experience_name` | string | Role title, organization name, event name, or certification name depending on type. |
| `organization` | string | Host organization. 'UMBC' or a campus unit for on-campus activities; an employer or issuing body otherwise. |
| `industry` | string | Industry of the host organization, or 'Not Applicable' for campus activities. |
| `term` | term | Term the activity took place, or in which a certification was earned. |
| `duration_terms` | int | Number of terms the activity lasted. 1 for single-term activities such as hackathons. |
| `hours_per_week` | int | Typical weekly hours. 'Not Applicable' for certifications. |
| `is_paid` | bool | TRUE if the activity was compensated. 'Not Applicable' for certifications. Values: `TRUE`, `FALSE`, `Not Applicable`. |
| `role_level` | enum | Level held within the activity. Values: `Member`, `Officer`, `President`, `Lead`, `Participant`, `Researcher`, `Employee`, `Mentor`, `Tutor`, `Not Applicable`. |
| `outcome` | string | How the activity concluded. Values vary by experience_type; see the notes below. |

## Sample rows

| `record_id` | `campus_id` | `experience_type` | `experience_name` | `organization` | `industry` | `term` | `duration_terms` | `hours_per_week` | `is_paid` | `role_level` | `outcome` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0000001 | CID-116490 | Internship | AI Research Intern | Federal Health Data Office | Federal Government | Fall 2026 | 1 | 40 | TRUE | Not Applicable | Completed |
| EXP-0000002 | CID-116490 | Certification | Project Management Associate | Cadence Project Institute | Not Applicable | Fall 2026 | 1 | Not Applicable | Not Applicable | Not Applicable | Earned |
| EXP-0000003 | CID-116490 | Certification | Machine Learning Associate | Applied Intelligence Institute | Not Applicable | Fall 2026 | 1 | Not Applicable | Not Applicable | Not Applicable | Earned |

## Gotchas

- Fields that do not apply to a given `experience_type` carry the literal `Not Applicable` rather than a blank. `hours_per_week` and `is_paid` are `Not Applicable` for every `Certification` row.
- `outcome` values differ by `experience_type`. Check which values occur for the type you are analyzing rather than assuming a shared vocabulary.
- This file covers current students and alumni together. Join to whichever person file you need, or to both.
- One person can hold several rows of the same type, including repeats of the same organization in different terms.

## Five questions this file alone can answer

- Which experience_type is most common, and how does that vary by major?
- What share of internships end in each outcome value?
- How many terms does the typical student organization membership last?
- Which organizations host the most internships?
- How does hours_per_week differ between paid and unpaid experiences?
