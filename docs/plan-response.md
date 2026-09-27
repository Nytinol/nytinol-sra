# Reading a `POST /plan` response

This is the contract for rendering a career plan. The payload is a ranked tree of named next steps, plus the goal the server understood and a snapshot of the student. Every number is a model estimate or a count from the synthetic HackUMBC dataset. Present alumni ids and outcome shares as records from that dataset.

Interactive schema: `GET /docs` while the server is running. Job families, industries, and seniority labels for a form come from `GET /reference`.

## Shape

```text
PlanResponse
  goal                         what the server mapped the free-text job onto
  student                      standing after the request was applied
  baseline_prob_goal_family    chance of the goal job family before any step
  recommendations[]            alternatives for the next term, best first
    next_steps[]               alternatives for the term after that step
      next_steps[]             one level deeper, until depth runs out
```

`recommendations` are siblings. They are different ideas for the same semester, already sorted. Render them in array order.

`next_steps` are also siblings, one semester later, and only make sense if the parent step was taken. A path is parent, then one child, then one grandchild. Do not lay the whole tree out as one ordered to-do list.

`width` on the request (default 3, max 5) is how many siblings to return at a level. `depth` (default 3, max 4) is how many semesters deep the tree goes. The last level has `"next_steps": []`.

## Header

### `goal`

The server turns `job_title` into fields that exist in the data. Show both the words the user typed and the mapped fields, so a loose match is visible.

| Field | Meaning | UI |
| --- | --- | --- |
| `requested_job_title` | The title that was sent | Quote it as the user's words |
| `job_family` | Mapped career area, such as `Cybersecurity` | This is what every "chance of reaching the goal" number uses |
| `industry` | Mapped industry, or `null` | `null` means the user left it blank or the text did not match an industry in the data. Hide industry targets in that case |
| `seniority` | Read from words in the title (`Senior`, `Lead`, …). Defaults to `Entry` | Label, not a score |
| `salary_usd` | Target salary in dollars, or `null` | `null` means no salary target was sent |

A title that matches nothing returns **422** with `{ "detail": "Could not map '<title>' onto a job family. ..." }`. That string lists the known families.

### `student`

This is the record the models actually scored, after classes and experiences were applied. Use it as the "you are here" panel.

| Field | Meaning |
| --- | --- |
| `class_level` | `Freshman`, `Sophomore`, `Junior`, or `Senior`, from credits |
| `credits_earned` | Credits used for that standing. A caller-supplied total wins over the class list |
| `gpa` | Reported GPA, or `null` |
| `skills_held` | Skills implied by the classes and experiences sent in |
| `internships_completed` | Internships plus co-ops already on the record |

### `baseline_prob_goal_family`

A probability from 0 to 1: how often students who look like this, before any recommended step, end up in `goal.job_family`.

Show it as the "before" number next to each step's `prob_reaches_goal_family`. Example: baseline `0.1389` and a step at `0.4867` is "14% today, about 49% after this step."

This number is only about job family. It does not include industry or salary.

## A recommendation card

Each object in `recommendations` and in any `next_steps` has the same fields.

### What to do

| Field | Meaning | UI |
| --- | --- | --- |
| `action_type` | `Course`, `Internship`, `Co-op`, `Certification`, or a campus activity type | Badge |
| `position` | Exact name: a course title, internship title, or certificate name | Card title |
| `industry` | Employer industry, or `null` | Show only when non-null. Certificates and courses are `null` |
| `target_term` | Semester this step is aimed at, such as `Fall 2026` | Put it on the card. Deeper cards move forward a semester |

Title line when `industry` is set: `{position} · {industry}`. When it is null: `{position}` only.

### Two different confidences

These must stay on separate labels. Combining them into one percent overstates how sure the industry call is.

| Field | Range | Meaning | UI |
| --- | --- | --- | --- |
| `feasibility` | 0 to 1 | How plausible it is that a student like this takes **this kind of step** next. For internships the score is summed across industries of the same role, so it answers "a cybersecurity internship," not "a cybersecurity internship at a bank" | Primary confidence for the role or course. A course often sits near `0.90` because the curriculum is predictable. An internship near `0.20`–`0.35` can still be the right suggestion |
| `industry_confidence` | 0 to 1, or `null` | Share of that role's plausibility that landed on **this** industry. `null` when the step has no industry | Secondary label, only when non-null. Expect low values. The model is weak at industry |

Copy that matches the numbers:

- Feasibility: "Fits students at this point"
- Industry confidence: "Less sure about the industry"

### Did it move the goal

| Field | Range | Meaning | UI |
| --- | --- | --- | --- |
| `prob_reaches_goal_family` | 0 to 1 | Chance of landing in `goal.job_family` **after** this step | The main outcome number. Compare it with `baseline_prob_goal_family` on the first level, and with the parent card's value deeper in the tree |
| `prob_reaches_goal_industry` | 0 to 1, or `null` | Chance of the goal industry after this step. `null` when the user did not name an industry the server could match | Show beside the family chance, with its own label. It is often much smaller. That is the model saying the role got closer and the industry did not |
| `goal_alignment` | 0 to 1 | Blended score after the step: job family, industry if one was set, and clearing the target salary if one was set. Weights shift when the user left industry or salary blank | A single "goal fit" meter. Do not caption it as the chance of getting the job |
| `goal_uplift` | signed decimal | `goal_alignment` after the step minus `goal_alignment` of the state before the step | Delta on the card: "goal fit +0.15". Later steps often show a small delta because the earlier step already moved the score. A negative value means the step pulls away from the goal; the ranker prefers positive deltas, but the field can still be negative |

`goal_alignment` stays well below 1 for an ambitious target (senior role, high salary, specific industry). That is expected. Lead with `prob_reaches_goal_family`, and use alignment as the combined fit.

### Salary

`projected_salary` is a band for a **first job**, in 2026 dollars. Integers.

| Field | Meaning |
| --- | --- |
| `p10` | Low end. About 1 in 10 similar outcomes fall below this |
| `median` | Middle guess |
| `p90` | High end. About 1 in 10 fall above this |

Render as `$73,540 – $97,454 – $118,066` or a range bar from `p10` to `p90` with the median marked. Compare `median` to `goal.salary_usd` when that field is set. The band is a first-job estimate, so a "senior" goal salary often sits near `p90` or above the band.

### Skills

| Field | Meaning | UI |
| --- | --- | --- |
| `skills_added` | Skills this step adds that the record does not already have | Chips. An empty list means this step teaches nothing new beyond the path so far; say "No new skills" rather than hiding the section |
| `skills_still_missing` | Skills the target role still wants after this step | A separate "still needed for {job_family}" list. It can stay long. That means one step does not finish the role |

### Reasons

`why` is a short list of plain phrases for why the next-step model liked this card ("the skills it builds are the ones the target role asks for").

Only the top level (`recommendations`) is filled in. Nested `next_steps` come back as `"why": []`. Hide the section when the list is empty.

### Evidence

Counts of people in the dataset who did this step and whose outcome is known. Same-major students are preferred.

| Field | Meaning | UI |
| --- | --- | --- |
| `alumni_count` | People with a known outcome who did this | "71 students" |
| `similar_alumni_count` | Same major, when that group is non-empty. Otherwise it falls back to the same people as `alumni_count` | "44 in Computer Science" |
| `basis` | Which group the shares describe | See below |
| `example_campus_ids` | Up to five ids from the similar group, such as `CID-945641` | Optional "for example" list. They are dataset ids |
| `median_first_salary_usd` | Median first salary of the similar group, 2026 dollars, or `null` | Separate from `projected_salary`. One is history, the other is the model's forecast for this student |
| `top_job_families` | Up to three `[name, share]` pairs. Share is 0 to 1, three decimals | Bar list. Names can include `Continuing Education` and `Still Seeking` |
| `top_industries` | Same shape, for employer industry | Bar list. The user's requested industry may be absent. Show the list anyway |
| `goal_family_share` | Share of the similar group whose outcome was `goal.job_family`, or `null` if there was no group | "48% went into Cybersecurity" |

`basis` is one of two strings:

- `"this role and industry"` — counts are for this exact step (this course, this certificate, or this role at this industry).
- `"this role across all industries"` — fewer than 10 same-major people did the exact pair, so the counts were widened to the role in every industry. Say so on the card. The internship title is still specific; the industry on the card is the suggestion, and the evidence is broader.

`goal_family_share` and `prob_reaches_goal_family` answer different questions. The share is "of people who already did this, how many landed in the field." The probability is "if this student does it next, what does the model expect." They often agree. When a very common course has a high `feasibility` and a low `goal_family_share`, show both: the course is a normal next class, and it is not what points people at the goal.

## Suggested layout

1. Header: mapped goal, student standing, baseline family chance.
2. Three columns, or a ranked list, for `recommendations`. Best first.
3. On each card: term, type, title, industry if present, family chance versus baseline, feasibility, industry confidence if present, salary band, skills added, evidence with `basis`.
4. Under the card, the `next_steps` as "Then, in {target_term}" alternatives. Repeat once more for the third semester.
5. `why` only on the first row of cards.

## Formatting

- Probabilities (`baseline_prob_goal_family`, `feasibility`, `industry_confidence`, `prob_reaches_goal_family`, `prob_reaches_goal_industry`, `goal_alignment`, evidence shares) are 0 to 1. Multiply by 100 for display. `feasibility` is rounded to 4 decimals; evidence shares are rounded to 3; `industry_confidence` is unrounded.
- Money fields are whole dollars.
- `goal_uplift` is a score delta, not a probability. Show the sign and two decimals (`+0.15`), or omit it if the family-chance comparison is already on the card.

## Empty and error states

| Situation | What you get |
| --- | --- |
| No legal next steps | `"recommendations": []` |
| End of the tree | `"next_steps": []` |
| No industry on the step | `industry: null`, `industry_confidence: null` |
| No industry on the goal | `goal.industry: null`, `prob_reaches_goal_industry: null` |
| Nested card | `why: []` |
| Step adds no new skills | `skills_added: []` |
| Job title matches no family | HTTP 422, `detail` string |
| Validation error (bad types, `width` or `depth` out of range) | HTTP 422, FastAPI's usual `detail` array |
