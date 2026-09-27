"""The shared skill vocabulary, and a skill profile for every course, activity,
credential and role.

`skill_tags` in the course catalog and `role_skill_tags` in the employment
history are drawn from the same 119-skill vocabulary, so what a transcript
covered and what a role asks for are directly comparable with no mapping table.
Everything in this module reduces to a vector over that vocabulary, which is how
the models compare a student's coursework against an opportunity and against
their goal.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from . import data

# A role profile is the share of roles in a family asking for each skill. Three
# or four skills are near-universal and the rest rotate, so 0.25 is the point
# that separates real demand from incidental mentions.
DEMAND_THRESHOLD = 0.25

INTERNSHIP_TYPES = ("Internship", "Co-op")
CAMPUS_TYPES = (
    "Student Organization",
    "Hackathon",
    "Undergraduate Research",
    "Tutoring",
    "Campus Job",
    "Peer Mentor",
    "Competitive Team",
)


@lru_cache(maxsize=1)
def vocabulary() -> tuple[str, ...]:
    catalog = data.load("course_catalog")
    employment = data.load("employment_history")
    skills: set[str] = set()
    for value in catalog["skill_tags"]:
        skills.update(data.split_pipe(value))
    for value in employment["role_skill_tags"]:
        skills.update(data.split_pipe(value))
    return tuple(sorted(skills))


@lru_cache(maxsize=1)
def _skill_position() -> dict[str, int]:
    return {skill: position for position, skill in enumerate(vocabulary())}


def size() -> int:
    return len(vocabulary())


def vector(skills: object) -> np.ndarray:
    """A binary vector over the vocabulary. Accepts a list or a pipe string."""
    values = data.split_pipe(skills) if isinstance(skills, str) else list(skills or [])
    result = np.zeros(size(), dtype=np.float32)
    positions = _skill_position()
    for skill in values:
        position = positions.get(skill)
        if position is not None:
            result[position] = 1.0
    return result


def names(mask: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Inverse of `vector`: the skills a profile actually covers."""
    vocab = vocabulary()
    return [vocab[position] for position in np.flatnonzero(np.asarray(mask) >= threshold)]


@lru_cache(maxsize=1)
def course_profiles() -> dict[str, np.ndarray]:
    catalog = data.load("course_catalog")
    return {
        str(row.course_id): vector(row.skill_tags)
        for row in catalog.itertuples(index=False)
    }


@lru_cache(maxsize=1)
def _role_profiles() -> dict[tuple[str, str], np.ndarray]:
    """Mean skill demand for each (job family, seniority) actually observed."""
    employment = data.load("employment_history")
    accumulator: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
    for row in employment.itertuples(index=False):
        key = (str(row.job_family), str(row.seniority_level))
        current, count = accumulator.get(key, (np.zeros(size(), dtype=np.float32), 0))
        accumulator[key] = (current + vector(row.role_skill_tags), count + 1)
    return {key: total / max(count, 1) for key, (total, count) in accumulator.items()}


def role_profile(job_family: str, seniority: str = "Entry") -> np.ndarray:
    profiles = _role_profiles()
    key = (job_family, seniority)
    if key in profiles:
        return profiles[key]
    # Fall back to any seniority within the family, then to an empty profile.
    within_family = [
        profile for (family, _), profile in profiles.items() if family == job_family
    ]
    if within_family:
        return np.mean(within_family, axis=0)
    return np.zeros(size(), dtype=np.float32)


def _teaching_profile(job_family: str) -> np.ndarray:
    """Entry-level demand in a family, rescaled so its top skill is 1.0.

    `role_profile` is a mean over roles, so its values are fractions of roles
    asking for each skill. An action's profile has to live on the same 0-1 scale
    as a student's coursework skill vector, because `transition` adds them.
    """
    profile = role_profile(job_family, "Entry")
    peak = float(profile.max())
    if peak <= 1e-9:
        return profile
    return profile / peak


@lru_cache(maxsize=1)
def job_families() -> tuple[str, ...]:
    return tuple(sorted({family for family, _ in _role_profiles()}))


@lru_cache(maxsize=1)
def industries() -> tuple[str, ...]:
    """The industries an employer can belong to, shared by jobs and internships."""
    employment = data.load("employment_history")
    return tuple(sorted({str(value) for value in employment["employer_industry"]}))


@lru_cache(maxsize=1)
def seniority_levels() -> tuple[str, ...]:
    order = ["Entry", "Mid", "Senior", "Lead", "Manager", "Director"]
    observed = {level for _, level in _role_profiles()}
    return tuple(level for level in order if level in observed)


@lru_cache(maxsize=1)
def _activity_affinity() -> dict[str, tuple[str, float]]:
    """Which job family each named experience points toward, and how strongly.

    Derived from the data rather than hand-mapped: for everyone who did a given
    activity, take the most common job family of their first job. `lift` is how
    much that beats the base rate for the family, so an activity that predicts
    nothing (most clubs) scores near zero while `Business Systems Intern`, whose
    alumni land in IT Business & Product 86% of the time, scores high.
    """
    alumni = data.load("alumni")
    experience = data.load("student_experience")
    employed = alumni[alumni["first_job_family"] != data.NOT_APPLICABLE]
    family_of = dict(zip(employed["campus_id"], employed["first_job_family"]))
    base = employed["first_job_family"].value_counts(normalize=True).to_dict()

    counts: dict[str, dict[str, int]] = {}
    for row in experience.itertuples(index=False):
        family = family_of.get(row.campus_id)
        if family is None:
            continue
        counts.setdefault(str(row.experience_name), {})
        bucket = counts[str(row.experience_name)]
        bucket[family] = bucket.get(family, 0) + 1

    affinity: dict[str, tuple[str, float]] = {}
    for name, bucket in counts.items():
        total = sum(bucket.values())
        if total < 10:
            continue
        family = max(bucket, key=lambda key: bucket[key])
        share = bucket[family] / total
        baseline = base.get(family, 0.0)
        lift = 0.0 if baseline >= 1.0 else max(0.0, (share - baseline) / (1.0 - baseline))
        affinity[name] = (family, lift)
    return affinity


def activity_affinity(experience_name: str) -> tuple[str | None, float]:
    return _activity_affinity().get(str(experience_name), (None, 0.0))


@lru_cache(maxsize=1)
def _credential_tokens() -> dict[str, np.ndarray]:
    """Skills named directly in a certification title.

    `Advanced SQL Practitioner` covers SQL; `Cloud Security Associate` covers
    Cloud Security. Matched on word boundaries so short tokens do not match
    inside longer words.
    """
    experience = data.load("student_experience")
    credentials = experience[experience["experience_type"] == "Certification"]
    profiles: dict[str, np.ndarray] = {}
    for name in sorted({str(value) for value in credentials["experience_name"]}):
        matched = np.zeros(size(), dtype=np.float32)
        haystack = name.lower()
        for position, skill in enumerate(vocabulary()):
            pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
            if re.search(pattern, haystack):
                matched[position] = 1.0
        profiles[name] = matched
    return profiles


def credential_profile(name: str) -> np.ndarray:
    """A certification's skill profile: what its title names, plus a weighted
    share of the entry-level demand in the family its holders land in."""
    named = _credential_tokens().get(str(name), np.zeros(size(), dtype=np.float32))
    family, lift = activity_affinity(name)
    if family is None:
        return named
    return np.clip(named + lift * _teaching_profile(family), 0.0, 1.0)


def experience_profile(experience_name: str, experience_type: str) -> np.ndarray:
    """Skill profile for any non-course experience.

    An internship in a family teaches close to what entry-level roles in that
    family ask for. A club or hackathon carries a much weaker signal, and the
    empirical lift scales it down automatically.
    """
    if experience_type == "Certification":
        return credential_profile(experience_name)
    family, lift = activity_affinity(experience_name)
    if family is None:
        return np.zeros(size(), dtype=np.float32)
    weight = lift if experience_type in INTERNSHIP_TYPES else 0.5 * lift
    return np.clip(weight * _teaching_profile(family), 0.0, 1.0)


def coverage(have: np.ndarray, want: np.ndarray, threshold: float = DEMAND_THRESHOLD) -> float:
    """Share of the skills a target asks for that the student already covers."""
    wanted = np.asarray(want) >= threshold
    if not wanted.any():
        return 1.0
    held = np.asarray(have)[wanted] >= threshold
    return float(held.mean())


def missing(have: np.ndarray, want: np.ndarray, threshold: float = DEMAND_THRESHOLD) -> list[str]:
    wanted = np.asarray(want) >= threshold
    held = np.asarray(have) >= threshold
    return names((wanted & ~held).astype(float))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(left @ right / denominator)
