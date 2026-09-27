"""Turn the six tables into training matrices.

Two datasets come out of here, both built by walking each person's timeline
forward one term at a time:

* the next-step dataset, which asks "from this situation, was this the
  opportunity they took?" — positives are what actually happened next, negatives
  are sampled from everything else that was feasible at that moment;
* the outcome dataset, which pairs many snapshots of the same person with the
  single outcome they eventually reached.

Both are split by graduation year rather than at random. A random split would put
the same cohort on both sides and inflate every metric.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import actions as action_space
from . import data, skills, state as state_module, terms

TRAIN_MAX_YEAR = 2022
VALIDATION_YEARS = (2023, 2024)
TEST_YEARS = (2025, 2026)

NEGATIVES_PER_POSITIVE = 5
MAX_COURSE_POSITIVES_PER_TERM = 2
RANDOM_SEED = 17

CONTINUING_EDUCATION = "Continuing Education"
STILL_SEEKING = "Still Seeking"
MILITARY = "Military"
NON_EMPLOYMENT_OUTCOMES = (CONTINUING_EDUCATION, STILL_SEEKING, MILITARY)
NOT_EMPLOYED_INDUSTRY = "No Employer"
UNKNOWN_DESTINATION = "No Response"

CATEGORICAL_FEATURES = (
    "major",
    "track",
    "entry_type",
    "class_level",
    "term_season",
    "goal_family",
    "goal_industry",
    "action_type",
    "action_industry",
    "action_family",
    "action_seniority",
    "action_course_level",
    "action_requirement",
)


@dataclass
class Timeline:
    campus_id: str
    major: str
    track: str
    entry_type: str
    entry_term_index: int
    work_hours: float
    graduation_year: int | None = None
    graduation_term_index: int | None = None
    is_alumnus: bool = True
    courses: dict[int, list[tuple]] = field(default_factory=dict)
    experiences: dict[int, list[tuple]] = field(default_factory=dict)
    spells: list[tuple] = field(default_factory=list)

    @property
    def event_terms(self) -> list[int]:
        return sorted(set(self.courses) | set(self.experiences))


@lru_cache(maxsize=1)
def timelines() -> dict[str, Timeline]:
    """Every person's ordered history, built in a single pass over each table."""
    alumni = data.load("alumni")
    students = data.load("students_current")
    transcripts = data.load("transcripts")
    experience = data.load("student_experience")
    employment = data.load("employment_history")

    entry_terms = (
        transcripts.assign(index=transcripts["term"].map(terms.term_index))
        .groupby("campus_id")["index"]
        .min()
        .astype(int)
        .to_dict()
    )

    result: dict[str, Timeline] = {}
    for row in alumni.itertuples(index=False):
        graduation = terms.term_index(row.graduation_term)
        result[row.campus_id] = Timeline(
            campus_id=row.campus_id,
            major=str(row.major),
            track=str(row.track),
            entry_type=str(row.entry_type),
            entry_term_index=entry_terms.get(row.campus_id, graduation - 8),
            work_hours=float(row.work_hours_per_week_while_enrolled),
            graduation_year=int(row.graduation_year),
            graduation_term_index=graduation,
            is_alumnus=True,
        )
    for row in students.itertuples(index=False):
        result[row.campus_id] = Timeline(
            campus_id=row.campus_id,
            major=str(row.major),
            track=str(row.track),
            entry_type=str(row.entry_type),
            entry_term_index=entry_terms.get(row.campus_id, terms.term_index(row.entry_term)),
            work_hours=float(row.work_hours_per_week),
            is_alumnus=False,
        )

    course_profiles = skills.course_profiles()
    for row in transcripts.itertuples(index=False):
        timeline = result.get(row.campus_id)
        if timeline is None:
            continue
        index = terms.term_index(row.term)
        timeline.courses.setdefault(index, []).append(
            (
                str(row.course_id),
                str(row.subject),
                float(row.credits_attempted),
                str(row.grade),
                None if pd.isna(row.grade_points) else float(row.grade_points),
                course_profiles.get(str(row.course_id)),
            )
        )

    for row in experience.itertuples(index=False):
        timeline = result.get(row.campus_id)
        if timeline is None or not terms.is_term(row.term):
            continue
        index = terms.term_index(row.term)
        timeline.experiences.setdefault(index, []).append(
            (
                str(row.experience_type),
                str(row.experience_name),
                str(row.industry),
                str(row.role_level),
                str(row.outcome),
                None if pd.isna(row.is_paid) else bool(row.is_paid),
            )
        )

    ordered = employment.sort_values(["campus_id", "start_date"])
    for row in ordered.itertuples(index=False):
        timeline = result.get(row.campus_id)
        if timeline is None:
            continue
        timeline.spells.append(
            (
                terms.index_from_date(row.start_date),
                str(row.job_family),
                str(row.seniority_level),
                str(row.employer_industry),
                float(row.annual_salary_usd),
                bool(row.is_current),
            )
        )
    return result


def initial_state(timeline: Timeline) -> state_module.State:
    return state_module.State(
        campus_id=timeline.campus_id,
        major=timeline.major,
        track=timeline.track,
        entry_type=timeline.entry_type,
        term_index=timeline.entry_term_index,
        entry_term_index=timeline.entry_term_index,
        work_hours=timeline.work_hours,
    )


def _course_action_id(course_id: str) -> str:
    return f"{action_space.COURSE}|{course_id}"


def _experience_action_id(experience_type: str, name: str, industry: str) -> str:
    label = "-" if industry == data.NOT_APPLICABLE else industry
    return f"{experience_type}|{name}|{label}"


def _role_action_id(family: str, seniority: str, industry: str) -> str:
    return f"{action_space.ROLE}|{family}|{seniority}|{industry}"


def walk(timeline: Timeline):
    """Yield `(snapshot, action_ids_taken)` for each term something happened.

    The snapshot is the state at the *start* of the term, before any of that
    term's courses or activities have been folded in, which is exactly the
    situation a student is in when deciding what to do next.
    """
    current = initial_state(timeline)
    for term in timeline.event_terms:
        current.term_index = term
        taken: list[str] = []
        for course_id, *_ in timeline.courses.get(term, []):
            taken.append(_course_action_id(course_id))
        for experience_type, name, industry, *_ in timeline.experiences.get(term, []):
            taken.append(_experience_action_id(experience_type, name, industry))

        yield current.clone(), taken

        for course_id, subject, credits, grade, points, profile in timeline.courses.get(
            term, []
        ):
            state_module.apply_course(
                current, course_id, subject, credits, grade, points, profile
            )
        for entry in timeline.experiences.get(term, []):
            experience_type, name, industry, role_level, outcome, is_paid = entry
            state_module.apply_experience(
                current, experience_type, name, industry, role_level, outcome, is_paid
            )
        current.terms_enrolled += 1

    if timeline.graduation_term_index is not None:
        current.term_index = timeline.graduation_term_index
        current.graduated = True
    yield current.clone(), []


def final_academic_state(timeline: Timeline) -> state_module.State:
    """State at graduation, used as the base for post-graduation role steps."""
    snapshot = None
    for snapshot, _ in walk(timeline):
        pass
    assert snapshot is not None
    return snapshot


@lru_cache(maxsize=1)
def outcomes() -> dict[str, dict]:
    """Each alum's career outcome, taken at their first job.

    Three measurements in this data decided the targets:

    * job family never changes within a person -- every alum's current family
      equals their first job's family -- so "which family do they end up in" and
      "which family do they start in" are the same question, and the first job is
      the one a student can actually plan toward;
    * industry does change: only 52% of alumni still work in the industry of
      their first job, so a terminal-industry target is half noise, while
      internship industry predicts *first* industry at nearly three times the
      independent rate;
    * salary is nominal for its start year, so it is converted to 2026 dollars.
      First-job salary is entry-level for everybody by construction, which keeps
      it comparable across cohorts, whereas a current salary would mix a 2015
      graduate's decade of raises with a 2026 graduate's starting offer.

    `No Response` is dropped rather than treated as unemployment. The docs are
    explicit that it means the outcome is unknown, and 480 graduates carry it.
    """
    alumni = data.load("alumni")
    result: dict[str, dict] = {}
    spells_by_person: dict[str, list[tuple]] = {}
    for campus_id, timeline in timelines().items():
        if timeline.spells:
            spells_by_person[campus_id] = timeline.spells

    for row in alumni.itertuples(index=False):
        destination = str(row.first_destination)
        if destination == UNKNOWN_DESTINATION:
            continue

        salary_real = None
        if not pd.isna(row.first_job_annual_salary_usd):
            salary_real = data.to_real_dollars(
                float(row.first_job_annual_salary_usd), int(row.graduation_year)
            )

        spells = spells_by_person.get(row.campus_id)
        if spells:
            first = spells[0]
            family, seniority, industry = first[1], first[2], first[3]
        elif destination in NON_EMPLOYMENT_OUTCOMES:
            family, seniority, industry = destination, None, NOT_EMPLOYED_INDUSTRY
        else:
            continue

        result[row.campus_id] = {
            "campus_id": row.campus_id,
            "graduation_year": int(row.graduation_year),
            "outcome_family": family,
            "outcome_seniority": seniority,
            "outcome_industry": industry,
            "first_job_family": (
                None
                if str(row.first_job_family) == data.NOT_APPLICABLE
                else str(row.first_job_family)
            ),
            "salary_real": salary_real,
        }
    return result


def goal_for(campus_id: str) -> state_module.Goal | None:
    """The goal a person's own history reveals, used to train the ranker.

    At inference the student states a goal; at training time the closest
    equivalent is the outcome they were in fact heading toward.
    """
    outcome = outcomes().get(campus_id)
    if outcome is None or outcome["outcome_family"] in NON_EMPLOYMENT_OUTCOMES:
        return None
    return state_module.Goal(
        job_family=outcome["outcome_family"],
        industry=(
            None
            if outcome["outcome_industry"] == NOT_EMPLOYED_INDUSTRY
            else outcome["outcome_industry"]
        ),
        salary_usd=outcome["salary_real"],
        seniority=outcome["outcome_seniority"] or "Entry",
    )


def split_label(graduation_year: int | None) -> str:
    if graduation_year is None:
        return "current"
    if graduation_year <= TRAIN_MAX_YEAR:
        return "train"
    if graduation_year in VALIDATION_YEARS:
        return "validation"
    return "test"


def build_next_step_dataset(
    negatives_per_positive: int = NEGATIVES_PER_POSITIVE,
    seed: int = RANDOM_SEED,
    include_roles: bool = True,
) -> pd.DataFrame:
    rng = random.Random(seed)
    catalog = action_space.by_id()
    pre_graduation = [
        action for action in action_space.action_space() if not action.is_role
    ]
    role_actions = [action for action in action_space.action_space() if action.is_role]

    rows: list[dict] = []
    for campus_id, timeline in timelines().items():
        if not timeline.is_alumnus:
            continue
        goal = goal_for(campus_id)
        split = split_label(timeline.graduation_year)

        for snapshot, taken_ids in walk(timeline):
            if not taken_ids:
                continue
            positives = [
                catalog[action_id] for action_id in taken_ids if action_id in catalog
            ]
            if not positives:
                continue
            courses = [action for action in positives if action.is_course]
            others = [action for action in positives if not action.is_course]
            rng.shuffle(courses)
            chosen = others + courses[:MAX_COURSE_POSITIVES_PER_TERM]

            feasible = [
                action
                for action in pre_graduation
                if action_space.feasible(action, snapshot)
            ]
            if not feasible:
                continue
            taken_set = set(taken_ids)
            pool = [action for action in feasible if action.action_id not in taken_set]

            for action in chosen:
                rows.append(
                    _next_step_row(snapshot, action, goal, 1, campus_id, split)
                )
                for negative in _sample(rng, pool, negatives_per_positive):
                    rows.append(
                        _next_step_row(snapshot, negative, goal, 0, campus_id, split)
                    )

        if include_roles and timeline.spells:
            rows.extend(
                _role_rows(timeline, goal, split, role_actions, rng, negatives_per_positive)
            )

    return to_frame(rows)


def _sample(rng: random.Random, pool: list, count: int) -> list:
    if len(pool) <= count:
        return list(pool)
    return rng.sample(pool, count)


def _next_step_row(
    snapshot: state_module.State,
    action: action_space.Action,
    goal: state_module.Goal | None,
    label: int,
    campus_id: str,
    split: str,
) -> dict:
    row = state_module.state_features(snapshot)
    row.update(state_module.goal_features(snapshot, goal))
    row.update(state_module.action_features(snapshot, action, goal))
    row["label"] = label
    row["campus_id"] = campus_id
    row["split"] = split
    row["group"] = f"{campus_id}:{snapshot.term_index}"
    row["action_id"] = action.action_id
    return row


def _role_rows(
    timeline: Timeline,
    goal: state_module.Goal | None,
    split: str,
    role_actions: list,
    rng: random.Random,
    negatives_per_positive: int,
) -> list[dict]:
    """Post-graduation steps: each job move, given the state after the last one."""
    catalog = action_space.by_id()
    current = final_academic_state(timeline)
    rows: list[dict] = []

    for index, spell in enumerate(timeline.spells):
        term, family, seniority, industry, _, _ = spell
        action_id = _role_action_id(family, seniority, industry)
        action = catalog.get(action_id)
        current.term_index = term

        if action is not None:
            snapshot = current.clone()
            feasible = [
                candidate
                for candidate in role_actions
                if action_space.feasible(candidate, snapshot)
                and candidate.action_id != action_id
            ]
            if feasible:
                rows.append(
                    _next_step_row(
                        snapshot, action, goal, 1, timeline.campus_id, split
                    )
                )
                for negative in _sample(rng, feasible, negatives_per_positive):
                    rows.append(
                        _next_step_row(
                            snapshot, negative, goal, 0, timeline.campus_id, split
                        )
                    )

        current.graduated = True
        current.seniority = seniority
        current.job_family = family
        current.job_industry = industry
        np.maximum(
            current.skill_vector,
            skills.role_profile(family, seniority),
            out=current.skill_vector,
        )
        if index == 0:
            current.completed_experiences.add(f"{action_space.ROLE}|{family}")
    return rows


def build_outcome_dataset(snapshot_stride: int = 1) -> pd.DataFrame:
    """One row per (person, term) snapshot, labelled with where they ended up.

    Snapshotting every term rather than only at graduation is deliberate: a
    freshman needs the same prediction a senior does, so the model has to have
    seen early states during training.
    """
    labels = outcomes()
    rows: list[dict] = []

    for campus_id, timeline in timelines().items():
        if not timeline.is_alumnus:
            continue
        outcome = labels.get(campus_id)
        if outcome is None:
            continue
        split = split_label(timeline.graduation_year)

        snapshots = [snapshot for snapshot, _ in walk(timeline)]
        for position, snapshot in enumerate(snapshots):
            if position % snapshot_stride:
                continue
            row = state_module.state_features(snapshot)
            row["campus_id"] = campus_id
            row["graduation_year"] = outcome["graduation_year"]
            row["split"] = split
            row["outcome_family"] = outcome["outcome_family"]
            row["outcome_industry"] = outcome["outcome_industry"]
            row["salary_real"] = outcome["salary_real"]
            row["progress"] = position / max(len(snapshots) - 1, 1)
            rows.append(row)

    return to_frame(rows)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "label",
        "campus_id",
        "split",
        "group",
        "action_id",
        "graduation_year",
        "outcome_family",
        "outcome_industry",
        "salary_real",
        "progress",
    }
    return [column for column in frame.columns if column not in excluded]


def align_categories(frame: pd.DataFrame, reference: dict[str, list]) -> pd.DataFrame:
    """Force categorical levels to match what a model was trained on."""
    frame = frame.copy()
    for column, categories in reference.items():
        if column in frame.columns:
            frame[column] = pd.Categorical(
                frame[column].astype(object), categories=categories
            )
    return frame


def category_levels(frame: pd.DataFrame) -> dict[str, list]:
    return {
        column: list(frame[column].cat.categories)
        for column in frame.columns
        if str(frame[column].dtype) == "category"
    }


def cached(name: str, builder, rebuild: bool = False) -> pd.DataFrame:
    """Build a dataset once and reuse it. Pickle, not CSV, to keep the
    categorical levels and the distinction between NaN and a missing category."""
    data.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = data.DERIVED_DIR / f"{name}.pkl"
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    frame = builder()
    frame.to_pickle(path)
    return frame


def next_step_dataset(rebuild: bool = False) -> pd.DataFrame:
    return cached("next_step", build_next_step_dataset, rebuild)


def outcome_dataset(rebuild: bool = False) -> pd.DataFrame:
    return cached("outcome", build_outcome_dataset, rebuild)


if __name__ == "__main__":
    outcome_frame = build_outcome_dataset()
    print("outcome dataset", outcome_frame.shape)
    print(outcome_frame["split"].value_counts().to_string())
    print(outcome_frame["outcome_family"].value_counts().to_string())

    outcome_frame.to_pickle(data.DERIVED_DIR / "outcome.pkl")

    next_step = build_next_step_dataset()
    print("\nnext-step dataset", next_step.shape)
    print(next_step["split"].value_counts().to_string())
    print(next_step["label"].value_counts().to_string())
    next_step.to_pickle(data.DERIVED_DIR / "next_step.pkl")
