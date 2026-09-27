# `employment_history.csv`

One row per job spell. Alumni only; current students have no employment records.

**Rows:** 6,028  ·  **Columns:** 19

## How it joins

`campus_id` joins to `alumni.csv` only.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `job_id` | string | Unique spell identifier, format JOB-NNNNNNN. |
| `campus_id` | string | Person this spell belongs to. Joins to alumni.csv. |
| `employer` | string | Employer name. All employers are fictitious. |
| `employer_industry` | string | Industry of the employer, denormalized. |
| `employer_size` | enum | Headcount band: Startup <100, Small 100-999, Mid 1k-10k, Large 10k-50k, Enterprise 50k+. Values: `Startup`, `Small`, `Mid`, `Large`, `Enterprise`. |
| `job_title` | string | Title held during this spell. |
| `job_family` | string | Broad role family the title belongs to. |
| `seniority_level` | enum | Seniority of the role. Values: `Entry`, `Mid`, `Senior`, `Lead`, `Manager`, `Director`. |
| `region` | string | Metro area of the role. |
| `cost_of_living_index` | int | Cost of living for that region, 100 = national average. Inline so real-terms comparisons need no join. |
| `is_remote` | bool | TRUE if the role was fully remote. Values: `TRUE`, `FALSE`. |
| `start_date` | date | First day of the spell, YYYY-MM-DD. |
| `end_date` | date | Last day of the spell, YYYY-MM-DD. Blank if and only if is_current is TRUE. This is the only blank cell in the dataset. |
| `is_current` | bool | TRUE if this is the person's current job as of September 2026. Values: `TRUE`, `FALSE`. |
| `tenure_months` | int | Length of the spell in months. For current jobs, months elapsed as of September 2026. |
| `annual_salary_usd` | int | Annual base salary in nominal dollars of the start year. |
| `change_type` | enum | How this spell began relative to the previous one. Values: `First Job`, `Internal Promotion`, `Internal Lateral`, `New Employer - Advance`, `New Employer - Lateral`, `Industry Switch`, `Return to School`, `Career Break`. |
| `requires_clearance` | bool | TRUE if the role required a security clearance. Values: `TRUE`, `FALSE`. |
| `role_skill_tags` | list | Pipe-delimited skills this role called for. Drawn from the same vocabulary as skill_tags in course_catalog.csv, so what a transcript covered and what a role asks for can be compared directly. |

## Sample rows

| `job_id` | `campus_id` | `employer` | `employer_industry` | `employer_size` | `job_title` | `job_family` | `seniority_level` | `region` | `cost_of_living_index` | `is_remote` | `start_date` | `end_date` | `is_current` | `tenure_months` | `annual_salary_usd` | `change_type` | `requires_clearance` | `role_skill_tags` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| JOB-0000001 | CID-655977 | Clearfield Assurance | Cybersecurity Services | Startup | IT Analyst | IT Business & Product | Entry | Northern Virginia | 114 | FALSE | 2021-01-15 | 2022-11-15 | FALSE | 23 | 84500 | First Job | FALSE | Business Process|Presentation|Product Management|Project Management|Requirements|Roadmapping|Technical Writing |
| JOB-0000002 | CID-655977 | Clearfield Assurance | Cybersecurity Services | Startup | Systems Analyst | IT Business & Product | Mid | Northern Virginia | 114 | FALSE | 2022-11-15 | 2025-03-15 | FALSE | 29 | 95750 | Internal Promotion | FALSE | Business Process|Presentation|Product Management|Project Management|Requirements|Roadmapping |
| JOB-0000003 | CID-655977 | Civic Code Alliance | Nonprofit & Civic Tech | Small | Senior Systems Analyst | IT Business & Product | Senior | Baltimore, MD | 100 | TRUE | 2025-03-15 | 2026-05-15 | FALSE | 15 | 103500 | New Employer - Advance | FALSE | Business Process|IT Governance|Management|Project Management|Requirements|Roadmapping|Strategy|User Research |

## Gotchas

- `end_date` is blank if and only if `is_current` is `TRUE`. This is the only blank cell in the dataset.
- Only alumni appear in this file. Current students have no employment records.
- Not every alum appears: those whose `first_destination` is `Continuing Education`, `Military`, `Still Seeking`, or `No Response` have no rows here.
- Spells for one person do not overlap, but there can be gaps between them.
- `annual_salary_usd` is nominal for the spell's start year and does not change within a spell.
- `role_skill_tags` is pipe-delimited and draws from the same vocabulary as `skill_tags` in `course_catalog.csv`, so the two can be compared with set operations directly. Skills vary by job family and seniority: a senior role asks for more than an entry-level one in the same family.

## Five questions this file alone can answer

- Which employers appear most often as a first job?
- How does annual_salary_usd vary by seniority_level and region?
- What is median tenure_months by employer_industry?
- How often does each change_type occur, and at what seniority?
- How does salary compare once divided by cost_of_living_index?
- Which skills appear most often in each job family, and how does that change with seniority?
