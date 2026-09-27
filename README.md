# Career Pathways Planner

Given where a student is now and the job they want, name the next opportunity to
go after — the exact role and industry — then show what it opens up after that.

Two models and a search loop, over the HackUMBC 2026 career-pathways dataset.
**The data is synthetic.** Nothing below is a fact about real UMBC graduates.

## How it works

For every opportunity still open to a student, two models are asked different
questions and the answers are combined:

| | question | model |
| --- | --- | --- |
| feasibility | could this student plausibly get this next? | Model A, a LightGBM relevance ranker over (state, action) pairs |
| uplift | does having done it move them closer to the goal? | Model B, LightGBM heads for job family, industry, and salary quantiles |

An opportunity that is easy to get but leads nowhere scores low, and so does one
that would be transformative but is out of reach. The winner is applied to the
student's state and the whole thing runs again, which is what produces a branching
roadmap rather than a single suggestion.

Model B is deliberately **not** conditioned on a goal. It learns the whole outcome
distribution from a state, and the student's particular goal is read off that
distribution afterwards. That avoids inventing goals for the 3,200 graduates who
never stated one.

## Running it

```bash
python -m model.train_outcome      # Model B, ~40s
python -m model.train_next_step    # Model A, ~80s
python -m model.evaluate           # every metric below, ~3 min
python -m uvicorn main:app         # POST /plan
python -m unittest discover -s tests
```

```bash
curl -X POST localhost:8000/plan -H 'content-type: application/json' -d '{
  "major": "Computer Science", "track": "Cybersecurity",
  "gpa": 3.1, "credits_earned": 68,
  "classes": ["CMSC201", "CMSC202", "CMSC203", "CMSC341", "MATH151"],
  "job_title": "senior cybersecurity analyst",
  "industry": "Financial Services", "salary": 115000,
  "current_term": "Fall 2026", "entry_term": "Fall 2024"
}'
```

For that student the top recommendation is the **Security Operations Analyst**
certification, which moves P(reaching Cybersecurity) from 0.14 to 0.50. The
supporting rows: 44 same-major alumni earned it, 48% of them went into
Cybersecurity against a 13% base rate, and their median first salary was $102,908.
It then branches into a **Cybersecurity Intern in Financial Services**.

## What the numbers say

Everything is measured on graduates of 2025-26; the models saw 2022 and earlier,
with 2023-24 held back for early stopping. The split is by graduation year, not at
random — classmates share courses, employers and a job market, so a random split
puts near-duplicates on both sides and inflates everything.

**Model A** ranks the full set of feasible options at a held-out decision point,
a median of 342 of them. Measured per opportunity actually taken:

| | recall@3 | MRR | vs. ranking by popularity |
| --- | --- | --- | --- |
| courses | 0.594 | 0.462 | 2.2x |
| internship role archetype | 0.612 | 0.556 | 2.0x |
| exact internship role *and* industry | 0.061 | 0.107 | — |
| certifications, campus activities | ~0.0 | ~0.05 | no better than popularity |

**Model B**, on 5,431 held-out snapshots: job family top-1 0.294 against a 0.178
majority class, and that rises from 0.219 to 0.393 between the first and last
quarter of a degree. Salary MAE $11,381 against a $12,624 median-guess baseline,
and the p10-p90 band covers 79.9% of held-out salaries against a nominal 80% —
the uncertainty is honest even though the point estimate is only slightly better
than guessing.

## What does not work

Worth saying plainly, because three of these shape the product:

- **Industry is close to unpredictable** from an academic record. The industry head
  reaches macro-F1 0.099. So a recommendation reports the *role* with its
  confidence and the *industry* separately with its own, much lower, confidence,
  and a stated industry preference is never overridden by the model.
- **Certifications and campus activities are not predictable next steps.** Model A
  does no better than popularity at guessing which club or credential someone
  takes next. It still recommends them, because Model B shows they move the
  outcome — but which specific one a student picks is not something this model
  knows.
- **The end-to-end backtest is underpowered.** Assigning each held-out graduate a
  goal at random and asking whether those who happened to take one of the
  planner's top three non-course suggestions reached it more often gives 16.1% vs
  9.8%, a 1.65x ratio — but the 95% interval is 0.56–3.11 and spans 1.0. Only 2.2%
  of held-out decisions coincided with a recommendation, which caps the power. The
  component models are validated; the end-to-end claim is not.
- It is observational throughout. Students who earn a certification differ from
  those who do not in ways the state does not capture, so none of this measures
  the effect of *following* the advice.

## Two findings from the data itself

- **Job family never changes.** Every graduate's current job family equals their
  first job's family, across all 2,147 with employment records. Industry does
  change — only 52% are still in their first one — which is why the industry head
  targets the first job and not the current one.
- **Salaries are flat in real terms.** Median first-job salary rose from $73,750 in
  2015 to $84,250 in 2026 nominally, but converted to 2026 dollars it goes from
  $87,819 to $84,250. Comparing cohorts without deflating measures inflation.

## Layout

```
data/raw/                  pristine dataset, read-only, the source of truth
data/derived/              typed and annotated copies; never written back to raw
model/terms.py             academic terms as a sortable integer
model/skills.py            the 119-skill vocabulary and a profile per opportunity
model/actions.py           657 named opportunities + hard feasibility constraints
model/state.py             as-of-term featuriser and transition(state, action)
model/datasets.py          timeline walking, negative sampling, temporal split
model/train_outcome.py     Model B
model/train_next_step.py   Model A
model/planner.py           beam search, SHAP reasons, alumni evidence
model/evaluate.py          every metric above
api.py                     free-text goal -> structured goal
main.py                    POST /plan
```

`data/raw/` is checked out read-only on purpose. An earlier cleanup script
filtered `role_level != "Not Applicable"` in place, which silently deleted all
4,106 internships, 678 co-ops and 2,845 certifications — the sentinel means the
field does not apply to those types, not that the row is bad. The scripts in
`Feature-engineering/` now read raw and write `data/derived/`, and refuse to write
into `data/raw/` at all.
