"""A student's situation at a point in time, and what taking an action does to it.

Everything here is computed *as of* a term. That matters more than it sounds:
`alumni.csv` carries `final_gpa`, `internship_count` and friends, but those are
end-of-degree totals. Feeding them into a state that represents a sophomore
leaks the future and produces a model that looks excellent and advises nothing.
So GPA, credits, skills and activity counts are all accumulated forward from the
transcript and experience rows instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from . import actions as action_space
from . import data, skills, terms

CREDITS_FOR_DEGREE = 120.0

ROLE_LEVEL_RANK = {
    "Not Applicable": 0,
    "Member": 1,
    "Participant": 1,
    "Employee": 1,
    "Mentor": 2,
    "Tutor": 2,
    "Researcher": 2,
    "Officer": 3,
    "Lead": 4,
    "President": 5,
}

# How well an activity went. `Completed` is the neutral middle; a return offer or
# a co-authored paper is a much stronger signal than mere attendance.
OUTCOME_QUALITY = {
    "Return offer accepted": 1.0,
    "Converted to part-time role": 0.8,
    "Return offer declined": 0.6,
    "Completed": 0.5,
    "Ended early": 0.05,
    "Grand prize": 1.0,
    "Category winner": 0.85,
    "Top 10 finish": 0.65,
    "Participated": 0.4,
    "Paper co-authored": 1.0,
    "Poster presented": 0.75,
    "Ongoing": 0.5,
    "National qualifier": 0.9,
    "Regional qualifier": 0.7,
    "Earned": 1.0,
    "Expired": 0.3,
    "Active": 0.5,
    "Left after one term": 0.15,
}

EXPERIENCE_TYPES = (
    "Internship",
    "Co-op",
    "Certification",
    "Student Organization",
    "Hackathon",
    "Undergraduate Research",
    "Tutoring",
    "Campus Job",
    "Peer Mentor",
    "Competitive Team",
)

CLASS_LEVELS = ("Freshman", "Sophomore", "Junior", "Senior")


def class_level(credits: float) -> str:
    if credits < 30:
        return "Freshman"
    if credits < 60:
        return "Sophomore"
    if credits < 90:
        return "Junior"
    return "Senior"


@lru_cache(maxsize=1)
def major_subjects() -> dict[str, str]:
    """Which transcript subject counts as 'the major' for each major.

    Derived from which subject dominates each major's core requirements rather
    than assumed, so `major_gpa` is reconstructed the same way the dataset
    computed it.
    """
    transcripts = data.load("transcripts")
    alumni = data.load("alumni")
    students = data.load("students_current")
    major_of = dict(zip(alumni["campus_id"], alumni["major"]))
    major_of.update(dict(zip(students["campus_id"], students["major"])))
    core = transcripts[transcripts["requirement_category"] == "Major Core"].copy()
    core["major"] = core["campus_id"].map(major_of)
    return (
        core.dropna(subset=["major"])
        .groupby("major")["subject"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )


@dataclass
class State:
    """Where a person stands at the start of `term_index`."""

    campus_id: str | None = None
    major: str = "Computer Science"
    track: str = "General"
    entry_type: str = "First-Time Freshman"
    term_index: int = 0
    entry_term_index: int = 0
    terms_enrolled: int = 0
    credits_earned: float = 0.0
    graded_credits: float = 0.0
    quality_points: float = 0.0
    major_graded_credits: float = 0.0
    major_quality_points: float = 0.0
    work_hours: float = 0.0
    skill_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(skills.size(), dtype=np.float32), repr=False
    )
    experience_counts: dict[str, int] = field(default_factory=dict)
    industries_touched: set[str] = field(default_factory=set)
    max_role_rank: int = 0
    outcome_quality_total: float = 0.0
    outcome_count: int = 0
    best_outcome_quality: float = 0.0
    paid_experiences: int = 0
    completed_courses: set[str] = field(default_factory=set)
    completed_experiences: set[str] = field(default_factory=set)
    # Strongest signal the student has accumulated toward each job family, and
    # the same restricted to internships and co-ops. Skill overlap alone loses
    # this: doing a Cybersecurity Intern says far more about where someone lands
    # than the handful of skills it happens to share with coursework.
    family_affinity: dict[str, float] = field(default_factory=dict)
    internship_affinity: dict[str, float] = field(default_factory=dict)
    graduated: bool = False
    seniority: str | None = None
    job_family: str | None = None
    job_industry: str | None = None

    def clone(self) -> "State":
        return replace(
            self,
            skill_vector=self.skill_vector.copy(),
            experience_counts=dict(self.experience_counts),
            industries_touched=set(self.industries_touched),
            completed_courses=set(self.completed_courses),
            completed_experiences=set(self.completed_experiences),
            family_affinity=dict(self.family_affinity),
            internship_affinity=dict(self.internship_affinity),
        )

    @property
    def term_offset(self) -> int:
        return self.term_index - self.entry_term_index

    @property
    def gpa(self) -> float | None:
        if self.graded_credits <= 0:
            return None
        return self.quality_points / self.graded_credits

    @property
    def major_gpa(self) -> float | None:
        if self.major_graded_credits <= 0:
            return None
        return self.major_quality_points / self.major_graded_credits

    @property
    def class_level(self) -> str:
        return class_level(self.credits_earned)

    @property
    def mean_outcome_quality(self) -> float:
        if self.outcome_count <= 0:
            return 0.0
        return self.outcome_quality_total / self.outcome_count

    @property
    def internship_count(self) -> int:
        return self.experience_counts.get("Internship", 0) + self.experience_counts.get(
            "Co-op", 0
        )


@dataclass(frozen=True)
class Goal:
    """What the student is aiming for."""

    job_family: str
    industry: str | None = None
    salary_usd: float | None = None
    job_title: str | None = None
    seniority: str = "Entry"

    @property
    def skill_profile(self) -> np.ndarray:
        return skills.role_profile(self.job_family, self.seniority)


def apply_course(
    state: State,
    course_id: str,
    subject: str,
    credits: float,
    grade: str,
    grade_points: float | None,
    skill_profile: np.ndarray | None = None,
) -> None:
    """Fold one completed course attempt into `state`, in place."""
    profile = (
        skill_profile
        if skill_profile is not None
        else skills.course_profiles().get(course_id)
    )
    counts_for_gpa = grade not in ("W", "IP") and grade_points is not None

    if grade not in ("W", "F", "IP"):
        state.credits_earned += credits
        state.completed_courses.add(course_id)
        if profile is not None:
            np.maximum(state.skill_vector, profile, out=state.skill_vector)
    if counts_for_gpa:
        state.graded_credits += credits
        state.quality_points += credits * float(grade_points)
        if subject == major_subjects().get(state.major):
            state.major_graded_credits += credits
            state.major_quality_points += credits * float(grade_points)


def apply_experience(
    state: State,
    experience_type: str,
    experience_name: str,
    industry: object,
    role_level: object,
    outcome: object,
    is_paid: object = None,
    skill_profile: np.ndarray | None = None,
) -> None:
    """Fold one activity into `state`, in place."""
    state.experience_counts[experience_type] = (
        state.experience_counts.get(experience_type, 0) + 1
    )
    state.completed_experiences.add(f"{experience_type}|{experience_name}")
    if isinstance(industry, str) and industry != data.NOT_APPLICABLE:
        state.industries_touched.add(industry)
    state.max_role_rank = max(
        state.max_role_rank, ROLE_LEVEL_RANK.get(str(role_level), 0)
    )
    quality = OUTCOME_QUALITY.get(str(outcome))
    if quality is not None:
        state.outcome_quality_total += quality
        state.outcome_count += 1
        state.best_outcome_quality = max(state.best_outcome_quality, quality)
    if is_paid is True:
        state.paid_experiences += 1

    family, lift = skills.activity_affinity(experience_name)
    if family is not None and lift > 0:
        state.family_affinity[family] = max(state.family_affinity.get(family, 0.0), lift)
        if experience_type in skills.INTERNSHIP_TYPES:
            state.internship_affinity[family] = max(
                state.internship_affinity.get(family, 0.0), lift
            )

    profile = (
        skill_profile
        if skill_profile is not None
        else skills.experience_profile(experience_name, experience_type)
    )
    np.maximum(state.skill_vector, profile, out=state.skill_vector)


def expected_grade_points(state: State, difficulty: float) -> float:
    """What grade to assume for a course the student has not taken yet.

    Anchored on their own record and nudged by how demanding the course is
    relative to the catalog average.
    """
    baseline = state.gpa if state.gpa is not None else 3.0
    mean_difficulty = _mean_difficulty()
    return float(np.clip(baseline - 0.3 * (difficulty - mean_difficulty), 0.0, 4.0))


@lru_cache(maxsize=1)
def _mean_difficulty() -> float:
    return float(data.load("course_catalog")["difficulty_index"].mean())


def transition(state: State, action: action_space.Action) -> State:
    """The state the student would be in after taking `action`.

    Deterministic and optimistic-but-typical: the action is assumed to go the way
    it usually goes, which is what makes the planner's counterfactual comparison
    between two candidate actions meaningful.
    """
    result = state.clone()

    if action.is_role:
        result.graduated = True
        result.seniority = action.seniority
        result.job_family = action.job_family
        result.job_industry = action.industry
        np.maximum(result.skill_vector, action.skill_profile, out=result.skill_vector)
        result.term_index = terms.next_term(result.term_index)
        return result

    if action.is_course:
        apply_course(
            result,
            course_id=str(action.course_id),
            subject=str(action.subject or ""),
            credits=action.credits,
            grade="B",
            grade_points=expected_grade_points(state, action.difficulty),
            skill_profile=action.skill_profile,
        )
    else:
        apply_experience(
            result,
            experience_type=action.action_type,
            experience_name=action.position,
            industry=action.industry or data.NOT_APPLICABLE,
            role_level="Member",
            outcome="Completed",
            is_paid=action.typical_hours > 0,
            skill_profile=action.skill_profile,
        )

    result.terms_enrolled = state.terms_enrolled + 1
    result.term_index = terms.next_term(result.term_index)
    return result


def state_features(state: State) -> dict[str, object]:
    """Features describing the student alone, shared by both models."""
    vector = state.skill_vector
    features: dict[str, object] = {
        "major": state.major,
        "track": state.track,
        "entry_type": state.entry_type,
        "class_level": state.class_level,
        "term_offset": float(state.term_offset),
        "term_season": terms.season_of(state.term_index),
        "terms_enrolled": float(state.terms_enrolled),
        "credits_earned": float(state.credits_earned),
        "credit_share": float(state.credits_earned / CREDITS_FOR_DEGREE),
        "gpa": float(state.gpa) if state.gpa is not None else np.nan,
        "major_gpa": float(state.major_gpa) if state.major_gpa is not None else np.nan,
        "graded_credits": float(state.graded_credits),
        "work_hours": float(state.work_hours),
        "skill_breadth": float(vector.sum()),
        "skill_count": float((vector >= 0.5).sum()),
        "courses_completed": float(len(state.completed_courses)),
        "industries_touched": float(len(state.industries_touched)),
        "max_role_rank": float(state.max_role_rank),
        "mean_outcome_quality": float(state.mean_outcome_quality),
        "best_outcome_quality": float(state.best_outcome_quality),
        "paid_experiences": float(state.paid_experiences),
        "graduated": float(state.graduated),
        "seniority_rank": float(
            action_space.SENIORITY_RANK.get(state.seniority or "", -1)
        ),
    }
    for experience_type in EXPERIENCE_TYPES:
        key = experience_type.lower().replace(" ", "_").replace("-", "_")
        features[f"n_{key}"] = float(state.experience_counts.get(experience_type, 0))
    features["n_internship_total"] = float(state.internship_count)

    # How well the student's coursework already lines up with entry-level demand
    # in each family, and how much their activities point there. Together these
    # are what let the outcome model say where someone is heading before they
    # have any job history at all.
    for family in skills.job_families():
        key = family.lower().replace(" & ", "_").replace(" ", "_")
        features[f"fit_{key}"] = skills.coverage(
            vector, skills.role_profile(family, "Entry")
        )
        features[f"affinity_{key}"] = float(state.family_affinity.get(family, 0.0))
        features[f"intern_affinity_{key}"] = float(
            state.internship_affinity.get(family, 0.0)
        )
    affinities = state.family_affinity.values()
    features["affinity_peak"] = float(max(affinities)) if affinities else 0.0
    features["affinity_spread"] = float(len([v for v in affinities if v > 0.2]))

    # Which industries the student has already worked in, not just how many.
    # Interning somewhere is the single clearest statement of where you will end
    # up working, and a bare count throws that away.
    for industry in skills.industries():
        key = industry.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")
        features[f"touched_{key}"] = float(industry in state.industries_touched)
    return features


def goal_features(state: State, goal: Goal | None) -> dict[str, object]:
    if goal is None:
        return {
            "goal_family": None,
            "goal_industry": None,
            "goal_coverage": np.nan,
            "goal_missing": np.nan,
            "goal_cosine": np.nan,
            "goal_salary_real": np.nan,
        }
    profile = goal.skill_profile
    return {
        "goal_family": goal.job_family,
        "goal_industry": goal.industry,
        "goal_coverage": skills.coverage(state.skill_vector, profile),
        "goal_missing": float(len(skills.missing(state.skill_vector, profile))),
        "goal_cosine": skills.cosine(state.skill_vector, profile),
        "goal_salary_real": float(goal.salary_usd) if goal.salary_usd else np.nan,
    }


def action_features(
    state: State, action: action_space.Action, goal: Goal | None = None
) -> dict[str, object]:
    """Features for one (state, action) pair, including the interactions.

    The interaction terms are the point: without them the ranker can only learn
    which opportunities are popular, not which ones fit this particular student.
    """
    profile = action.skill_profile
    gain = np.clip(profile - state.skill_vector, 0.0, None)
    catalog_required = False
    if action.is_course and action.course_id:
        catalog = _course_requirements()
        catalog_required = state.major in catalog.get(action.course_id, ())

    features: dict[str, object] = {
        "action_type": action.action_type,
        "action_industry": action.industry,
        "action_family": action.job_family,
        "action_seniority": action.seniority,
        "action_course_level": action.course_level,
        "action_requirement": action.requirement_category,
        "action_observations": float(np.log1p(action.observations)),
        "action_credits": float(action.credits),
        "action_difficulty": float(action.difficulty),
        "action_duration_terms": float(action.duration_terms),
        "action_hours": float(action.typical_hours),
        "action_prereq_count": float(len(action.prerequisites)),
        "action_required_for_major": float(catalog_required),
        "action_term_slack": float(state.term_offset - action.earliest_term_offset),
        "skill_gain": float(gain.sum()),
        "skill_gain_count": float((gain >= 0.35).sum()),
        "skill_overlap": skills.cosine(state.skill_vector, profile),
        "industry_already_touched": float(
            action.industry in state.industries_touched if action.industry else 0.0
        ),
    }

    if goal is None:
        features.update(
            {
                "action_goal_cosine": np.nan,
                "goal_gap_closed": np.nan,
                "action_matches_goal_family": np.nan,
                "action_matches_goal_industry": np.nan,
            }
        )
    else:
        goal_profile = goal.skill_profile
        wanted_and_missing = np.clip(
            goal_profile - state.skill_vector, 0.0, None
        )
        features.update(
            {
                "action_goal_cosine": skills.cosine(profile, goal_profile),
                "goal_gap_closed": float(np.minimum(gain, wanted_and_missing).sum()),
                "action_matches_goal_family": float(
                    action.job_family == goal.job_family
                ),
                "action_matches_goal_industry": float(
                    goal.industry is not None and action.industry == goal.industry
                ),
            }
        )
    return features


@lru_cache(maxsize=1)
def _course_requirements() -> dict[str, tuple[str, ...]]:
    catalog = data.load("course_catalog")
    return {
        str(row.course_id): tuple(data.split_pipe(row.required_for_majors))
        for row in catalog.itertuples(index=False)
    }


def from_request(
    major: str,
    track: str,
    gpa: float | None,
    credits_earned: float,
    completed_courses: list[str],
    experiences: list[dict],
    current_term: str,
    entry_term: str | None = None,
    entry_type: str = "First-Time Freshman",
    work_hours: float = 0.0,
    campus_id: str | None = None,
) -> State:
    """Build a state from API input rather than from the dataset.

    Course ids may arrive as ids or titles, so both are accepted.
    """
    term_index = terms.term_index(current_term)
    catalog = data.load("course_catalog")
    by_title = {
        str(row.course_title).strip().lower(): str(row.course_id)
        for row in catalog.itertuples(index=False)
    }
    known_ids = {str(value) for value in catalog["course_id"]}

    resolved: set[str] = set()
    for entry in completed_courses:
        token = str(entry).strip()
        if token in known_ids:
            resolved.add(token)
        elif token.upper() in known_ids:
            resolved.add(token.upper())
        elif token.lower() in by_title:
            resolved.add(by_title[token.lower()])

    state = State(
        campus_id=campus_id,
        major=major,
        track=track or "General",
        entry_type=entry_type,
        term_index=term_index,
        entry_term_index=terms.term_index(entry_term) if entry_term else term_index - 6,
        work_hours=work_hours,
    )

    credit_by_course = dict(zip(catalog["course_id"], catalog["credits"]))
    subject_by_course = dict(zip(catalog["course_id"], catalog["subject"]))
    assumed_points = 3.0 if gpa is None else float(gpa)
    for course_id in resolved:
        apply_course(
            state,
            course_id=course_id,
            subject=str(subject_by_course.get(course_id, "")),
            credits=float(credit_by_course.get(course_id, 3)),
            grade="B",
            grade_points=assumed_points,
        )

    # Trust the reported GPA and credits over what the course list implies: the
    # caller knows their transcript, and the list may be partial.
    if gpa is not None:
        state.graded_credits = max(state.graded_credits, 1.0)
        state.quality_points = state.graded_credits * float(gpa)
        state.major_graded_credits = max(state.major_graded_credits, 1.0)
        state.major_quality_points = state.major_graded_credits * float(gpa)
    if credits_earned:
        state.credits_earned = float(credits_earned)
    state.terms_enrolled = max(1, state.term_offset)

    for entry in experiences:
        apply_experience(
            state,
            experience_type=str(entry.get("experience_type", "Student Organization")),
            experience_name=str(entry.get("experience_name", "")),
            industry=entry.get("industry") or data.NOT_APPLICABLE,
            role_level=entry.get("role_level", "Member"),
            outcome=entry.get("outcome", "Completed"),
            is_paid=entry.get("is_paid"),
        )
    return state
