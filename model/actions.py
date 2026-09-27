"""The action space: every opportunity the planner is allowed to recommend.

Every action is nameable, which is what lets a recommendation say the exact
position and industry rather than a vague direction. Actions come from what
people in the data actually did, so the space cannot contain an internship title
or a role that does not exist.

`feasible` applies the hard constraints. These are not soft scores: a course
whose prerequisites are unmet, or one not offered in the upcoming season, is
removed before any model sees it, so the output cannot be impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import data, skills, terms

# An opportunity has to appear this many times before it is recommendable. Rare
# one-offs are real, but they are not advice.
MIN_OBSERVATIONS = 5

COURSE = "Course"
ROLE = "Role"
SENIORITY_RANK = {level: rank for rank, level in enumerate(skills.seniority_levels())}


@dataclass(frozen=True)
class Action:
    action_id: str
    action_type: str
    position: str
    industry: str | None = None
    course_id: str | None = None
    subject: str | None = None
    job_family: str | None = None
    seniority: str | None = None
    credits: float = 0.0
    difficulty: float = 0.0
    course_level: str | None = None
    requirement_category: str | None = None
    terms_offered: tuple[str, ...] = ()
    prerequisites: tuple[tuple[str, ...], ...] = ()
    duration_terms: float = 1.0
    typical_hours: float = 0.0
    observations: int = 0
    earliest_term_offset: int = 0
    skill_profile: np.ndarray = field(
        default_factory=lambda: np.zeros(skills.size(), dtype=np.float32),
        compare=False,
        repr=False,
    )

    @property
    def label(self) -> str:
        if self.industry:
            return f"{self.position} ({self.industry})"
        return self.position

    @property
    def is_course(self) -> bool:
        return self.action_type == COURSE

    @property
    def is_role(self) -> bool:
        return self.action_type == ROLE


def _parse_prerequisites(value: object) -> tuple[tuple[str, ...], ...]:
    """Each element is a group of interchangeable alternatives.

    The catalog writes alternatives as `MATH151 or MATH155`, meaning either one
    satisfies the requirement.
    """
    groups = []
    for element in data.split_pipe(value):
        groups.append(tuple(part.strip() for part in element.split(" or ") if part.strip()))
    return tuple(groups)


@lru_cache(maxsize=1)
def _entry_terms() -> dict[str, int]:
    """First term each person has a transcript row for."""
    transcripts = data.load("transcripts")
    indices = transcripts["term"].map(terms.term_index)
    return indices.groupby(transcripts["campus_id"]).min().astype(int).to_dict()


@lru_cache(maxsize=1)
def _course_actions() -> list[Action]:
    catalog = data.load("course_catalog")
    transcripts = data.load("transcripts")
    counts = transcripts["course_id"].value_counts().to_dict()
    category = (
        transcripts.groupby("course_id")["requirement_category"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    offsets = _earliest_offsets()
    profiles = skills.course_profiles()

    actions = []
    for row in catalog.itertuples(index=False):
        course_id = str(row.course_id)
        actions.append(
            Action(
                action_id=f"{COURSE}|{course_id}",
                action_type=COURSE,
                position=f"{course_id} {row.course_title}",
                course_id=course_id,
                subject=str(row.subject),
                credits=float(row.credits),
                difficulty=float(row.difficulty_index),
                course_level=str(row.course_level),
                requirement_category=category.get(course_id),
                terms_offered=tuple(data.split_pipe(row.typical_terms_offered)),
                prerequisites=_parse_prerequisites(row.prerequisite_ids),
                observations=int(counts.get(course_id, 0)),
                earliest_term_offset=offsets.get(f"{COURSE}|{course_id}", 0),
                skill_profile=profiles.get(course_id, np.zeros(skills.size(), np.float32)),
            )
        )
    return actions


@lru_cache(maxsize=1)
def _earliest_offsets() -> dict[str, int]:
    """How far into a degree each opportunity first realistically appears.

    Taken as the 5th percentile of the term offsets at which people actually did
    it, so a capstone or a senior-level internship cannot be recommended to a
    first-semester freshman, and the cutoff comes from the data rather than a
    guess about class standing.
    """
    entry = _entry_terms()
    offsets: dict[str, list[int]] = {}

    transcripts = data.load("transcripts")
    for row in transcripts.itertuples(index=False):
        start = entry.get(row.campus_id)
        if start is None:
            continue
        offsets.setdefault(f"{COURSE}|{row.course_id}", []).append(
            terms.term_index(row.term) - start
        )

    experience = data.load("student_experience")
    for row in experience.itertuples(index=False):
        start = entry.get(row.campus_id)
        if start is None or not terms.is_term(row.term):
            continue
        key = _experience_key(str(row.experience_type), str(row.experience_name))
        offsets.setdefault(key, []).append(terms.term_index(row.term) - start)

    return {
        key: int(np.percentile(values, 5))
        for key, values in offsets.items()
        if values
    }


def _experience_key(experience_type: str, experience_name: str) -> str:
    return f"{experience_type}|{experience_name}"


@lru_cache(maxsize=1)
def _experience_actions() -> list[Action]:
    experience = data.load("student_experience")
    offsets = _earliest_offsets()

    grouped = experience.groupby(
        ["experience_type", "experience_name", "industry"], dropna=False
    ).agg(
        observations=("record_id", "size"),
        duration_terms=("duration_terms", "median"),
        typical_hours=("hours_per_week", "median"),
    )

    actions = []
    for (experience_type, experience_name, industry), row in grouped.iterrows():
        if int(row.observations) < MIN_OBSERVATIONS:
            continue
        industry_label = None if industry == data.NOT_APPLICABLE else str(industry)
        family, _ = skills.activity_affinity(experience_name)
        key = _experience_key(str(experience_type), str(experience_name))
        actions.append(
            Action(
                action_id=f"{experience_type}|{experience_name}|{industry_label or '-'}",
                action_type=str(experience_type),
                position=str(experience_name),
                industry=industry_label,
                job_family=family,
                duration_terms=float(row.duration_terms) if row.duration_terms else 1.0,
                typical_hours=0.0 if np.isnan(row.typical_hours) else float(row.typical_hours),
                observations=int(row.observations),
                earliest_term_offset=offsets.get(key, 0),
                skill_profile=skills.experience_profile(
                    str(experience_name), str(experience_type)
                ),
            )
        )
    return actions


@lru_cache(maxsize=1)
def _role_actions() -> list[Action]:
    employment = data.load("employment_history")
    grouped = employment.groupby(
        ["job_family", "seniority_level", "employer_industry"]
    ).agg(observations=("job_id", "size"), title=("job_title", lambda v: v.value_counts().idxmax()))

    actions = []
    for (job_family, seniority, industry), row in grouped.iterrows():
        if int(row.observations) < MIN_OBSERVATIONS:
            continue
        actions.append(
            Action(
                action_id=f"{ROLE}|{job_family}|{seniority}|{industry}",
                action_type=ROLE,
                position=str(row.title),
                industry=str(industry),
                job_family=str(job_family),
                seniority=str(seniority),
                observations=int(row.observations),
                skill_profile=skills.role_profile(str(job_family), str(seniority)),
            )
        )
    return actions


@lru_cache(maxsize=1)
def action_space() -> tuple[Action, ...]:
    return tuple(_course_actions() + _experience_actions() + _role_actions())


@lru_cache(maxsize=1)
def by_id() -> dict[str, Action]:
    return {action.action_id: action for action in action_space()}


def lookup(action_id: str) -> Action | None:
    return by_id().get(action_id)


def prerequisites_met(
    action: Action, completed_courses: set[str], entry_type: str = "First-Time Freshman"
) -> bool:
    """Whether the student has cleared the gate on a course.

    Transfer entrants get the benefit of the doubt. Their earlier coursework was
    done at another institution and so has no transcript rows at all, which would
    otherwise make every gated course look permanently out of reach -- it blocked
    a third of the real course enrolments of transfer students before this
    exception. A prerequisite they have not attempted here is assumed to have
    come in with them; first-time freshmen are still held to the full chain.
    """
    if entry_type == "Transfer":
        return True
    return all(
        any(alternative in completed_courses for alternative in group)
        for group in action.prerequisites
    )


def feasible(action: Action, state, target_term: int | None = None) -> bool:
    """Whether `action` could actually be taken next from `state`.

    Hard constraints only. Anything that survives this is a real option; ranking
    it is the models' job.
    """
    term = state.term_index if target_term is None else target_term

    if action.is_role:
        if not state.graduated:
            return False
        current = SENIORITY_RANK.get(state.seniority or "", -1)
        # You advance at most one rung at a time, and do not step backwards.
        return current <= SENIORITY_RANK.get(action.seniority or "Entry", 0) <= current + 1

    if state.graduated:
        return False

    if action.earliest_term_offset > state.term_offset:
        return False

    if action.is_course:
        if action.course_id in state.completed_courses:
            return False
        if action.terms_offered and terms.season_of(term) not in action.terms_offered:
            return False
        return prerequisites_met(action, state.completed_courses, state.entry_type)

    # A student does not repeat a named activity or re-earn a credential, but
    # may hold the same kind of internship at a different employer.
    return _experience_key(action.action_type, action.position) not in state.completed_experiences


def candidates(state, target_term: int | None = None) -> list[Action]:
    return [action for action in action_space() if feasible(action, state, target_term)]


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in action_space():
        counts[action.action_type] = counts.get(action.action_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))
