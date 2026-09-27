# `alumni.csv`

One row per graduate, highest degree only.

**Rows:** 320  ·  **Columns:** 29

## How it joins

`campus_id` joins to `transcripts.csv`, `student_experience.csv`, and `employment_history.csv`. It never appears in `students_current.csv`.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `campus_id` | string | Unique person identifier, format CID-NNNNNN. Joins to every other file. |
| `major` | enum | Major of the degree earned. Values: `Computer Science`, `Information Systems`. |
| `degree_level` | enum | Degree earned. Master's holders who also hold a UMBC bachelor's are flagged by holds_prior_umbc_bachelors. Values: `Bachelor of Science`, `Master of Science`. |
| `track` | string | Concentration within the major. |
| `graduation_term` | term | Term the degree was conferred. |
| `graduation_year` | int | Calendar year of graduation, 2015-2026. |
| `entry_type` | enum | Whether the student started here or transferred in. Values: `First-Time Freshman`, `Transfer`. |
| `time_to_degree_years` | float | Years from entry term to graduation term, one decimal. |
| `total_credits_earned` | int | Credits earned at UMBC toward the degree. |
| `final_gpa` | float | Cumulative GPA at graduation. |
| `major_gpa` | float | GPA across courses in the major subject only. |
| `residency` | enum | Tuition residency status while enrolled. Values: `In-State`, `Out-of-State`. |
| `holds_prior_umbc_bachelors` | bool | TRUE if this person also earned a UMBC bachelor's degree before this one. Values: `TRUE`, `FALSE`. |
| `work_hours_per_week_while_enrolled` | int | Average paid work hours per week during enrollment. |
| `internship_count` | int | Internship and co-op records in student_experience.csv. |
| `credential_count` | int | Certification records in student_experience.csv. |
| `engagement_activity_count` | int | All other experience records. |
| `net_cost_usd` | int | Total tuition and fees paid after grant and scholarship aid, across the whole degree. |
| `total_loans_usd` | int | Portion of net cost financed with loans. |
| `first_destination` | enum | Status approximately six months after graduation. Values: `Employed Full-Time`, `Employed Part-Time`, `Continuing Education`, `Military`, `Still Seeking`, `No Response`. |
| `months_to_first_job` | float | Months from graduation to the start of the first job. 'Not Applicable' where first_destination is not an employed category. |
| `first_job_title` | string | Title of the first job after graduation, or 'Not Applicable'. |
| `first_job_family` | string | Job family of the first job, or 'Not Applicable'. |
| `first_employer` | string | Employer of the first job, or 'Not Applicable'. |
| `first_employer_industry` | string | Industry of that employer, or 'Not Applicable'. |
| `first_job_region` | string | Metro area of the first job, or 'Not Applicable'. |
| `first_job_annual_salary_usd` | int | Annual base salary of the first job in nominal dollars of its start year. 'Not Applicable' where not employed. |
| `first_job_is_remote` | bool | TRUE if the first job was fully remote. 'Not Applicable' where not employed. Values: `TRUE`, `FALSE`, `Not Applicable`. |
| `first_job_found_via` | enum | How the first job was found. Values: `Return Offer from Internship`, `Career Fair`, `Online Application`, `Faculty or Staff Referral`, `Alumni Network`, `Personal Network`, `Recruiter Outreach`, `Student Organization Connection`, `Not Applicable`. |

## Sample rows

| `campus_id` | `major` | `degree_level` | `track` | `graduation_term` | `graduation_year` | `entry_type` | `time_to_degree_years` | `total_credits_earned` | `final_gpa` | `major_gpa` | `residency` | `holds_prior_umbc_bachelors` | `work_hours_per_week_while_enrolled` | `internship_count` | `credential_count` | `engagement_activity_count` | `net_cost_usd` | `total_loans_usd` | `first_destination` | `months_to_first_job` | `first_job_title` | `first_job_family` | `first_employer` | `first_employer_industry` | `first_job_region` | `first_job_annual_salary_usd` | `first_job_is_remote` | `first_job_found_via` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CID-655977 | Information Systems | Bachelor of Science | General | Fall 2020 | 2020 | First-Time Freshman | 7.7 | 121 | 2.44 | 2.50 | In-State | FALSE | 16 | 1 | 2 | 2 | 44020 | 28920 | Employed Full-Time | 1.8 | IT Analyst | IT Business & Product | Clearfield Assurance | Cybersecurity Services | Northern Virginia | 84500 | FALSE | Online Application |
| CID-861644 | Computer Science | Bachelor of Science | Software Engineering | Fall 2025 | 2025 | First-Time Freshman | 3.7 | 122 | 3.08 | 2.98 | In-State | FALSE | 10 | 0 | 1 | 1 | 41340 | 21710 | Employed Part-Time | 3.5 | Associate Software Engineer | Software Engineering | Tidal Power Cooperative | Energy & Utilities | Columbia/Howard County, MD | 74500 | FALSE | Online Application |
| CID-617792 | Computer Science | Bachelor of Science | Data Science | Spring 2022 | 2022 | First-Time Freshman | 4.0 | 122 | 2.57 | 2.75 | In-State | FALSE | 21 | 3 | 1 | 3 | 35700 | 26300 | No Response | Not Applicable | Not Applicable | Not Applicable | Not Applicable | Not Applicable | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

## Gotchas

- Where `first_destination` is not an employed category, every first-job column carries the literal `Not Applicable` rather than a blank.
- `No Response` is a real category covering about 15% of graduates. It means the outcome is unknown, not that the person was unemployed. Decide explicitly whether to exclude those rows before computing placement rates.
- `first_job_annual_salary_usd` is in nominal dollars of the year the job started. Comparing 2015 to 2026 without adjusting compares different dollars.
- `total_credits_earned` includes transferred credits for `Transfer` entrants, which do not appear in `transcripts.csv`.

## Five questions this file alone can answer

- How does first_job_annual_salary_usd differ across graduation years?
- Which first_job_found_via routes are most common, and do they differ by major?
- What is the spread of net_cost_usd by residency and entry_type?
- How long is months_to_first_job, and how does it vary by graduation year?
- What share of each graduating class reports each first_destination?
