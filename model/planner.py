"""The planner: turn two models into a named next step, then keep going.

For every opportunity still open to the student, ask the two models a different
question and combine the answers:

    feasibility  (Model A)  could this student plausibly get this next?
    uplift       (Model B)  does being the person who has done it move them
                            closer to the goal than they are now?

An opportunity that is easy to get but leads nowhere scores low, and so does one
that would be transformative but is out of reach. The chosen step is then applied
to the state and the whole thing runs again, which is what produces a branching
roadmap instead of a single suggestion.

Two parts of a recommendation are reported separately because they are not
equally trustworthy. Which *kind* of opportunity to pursue is a real prediction:
the ranker puts the internship archetype a student actually took in its top three
61% of the time, against 31% for ranking by popularity. Which *industry* to do it
in is much weaker, because in this data industry barely follows from anything
academic. So industry carries its own conditional confidence rather than being
folded into one number that would overstate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import actions as action_space
from . import data, datasets, skills, terms
from . import state as state_module
from .train_next_step import NextStepModel
from .train_outcome import OutcomeModel

# Shortlist by feasibility, then run the counterfactual only on what survives.
# Transitioning and re-scoring all ~350 open options at every node would dominate
# the response time without changing the answer.
#
# The quota per kind matters more than it looks. Ranking one global shortlist by
# feasibility fills it entirely with courses: a course scores around 0.98 because
# curriculum order is nearly deterministic, while each internship splits its
# probability across 16 industry variants and lands near 0.05. A student aiming at
# a machine-learning job would then never be shown a machine-learning internship.
QUOTAS = {"Course": 12, "Internship": 5, "Certification": 6, "other": 8}

# For each internship archetype that makes the cut, this many industry variants
# are costed out so the outcome model can pick the industry rather than the
# feasibility model, which measurably cannot.
INDUSTRY_VARIANTS = 8

BEAM_WIDTH = 3
BEAM_DEPTH = 3

# A step that is realistic but goal-neutral should still be able to surface, so
# feasibility carries weight on its own. Uplift is rescaled against the best
# candidate at the same node before being combined: a single term moves the
# outcome distribution by only a few thousandths, so on a raw scale it would be
# swamped by this floor and the ranking would collapse back to pure feasibility.
FEASIBILITY_BONUS = 0.15

# Penalty applied to a candidate repeating a kind of step already chosen, so the
# three recommendations are three different ideas rather than three courses.
DIVERSITY_PENALTY = 0.45

FEATURE_PHRASES = {
    "skill_gain": "adds skills the transcript does not cover yet",
    "skill_gain_count": "covers several skills that are currently missing",
    "skill_overlap": "builds directly on coursework already completed",
    "goal_gap_closed": "closes part of the skill gap to the target role",
    "action_goal_cosine": "the skills it builds are the ones the target role asks for",
    "action_matches_goal_family": "sits in the same job family as the goal",
    "action_matches_goal_industry": "sits in the target industry",
    "action_required_for_major": "required for the major",
    "action_requirement": "counts toward a degree requirement",
    "action_term_slack": "the timing fits where the student is in the degree",
    "action_observations": "a well-trodden route in this data",
    "action_difficulty": "the difficulty is in line with the student's record",
    "action_credits": "the credit load fits the student's pace",
    "action_type": "the kind of step students take at this stage",
    "action_course_level": "pitched at the right level for this student",
    "industry_already_touched": "the student has already worked in this industry",
    "credits_earned": "matches the student's credit standing",
    "credit_share": "matches the student's progress through the degree",
    "term_offset": "fits the point the student has reached",
    "terms_enrolled": "fits how long the student has been enrolled",
    "gpa": "consistent with the student's GPA",
    "major_gpa": "consistent with the student's GPA in the major",
    "n_internship_total": "reflects the internships already completed",
    "affinity_peak": "continues the direction the student's activities point in",
    "affinity_spread": "broadens a focus that is currently narrow",
    "class_level": "appropriate for the student's class standing",
    "courses_completed": "fits how much coursework is already done",
    "max_role_rank": "matches the level of responsibility already held",
    "best_outcome_quality": "builds on how well earlier activities went",
    "mean_outcome_quality": "builds on how well earlier activities went",
    "goal_coverage": "reflects how much of the target role is already covered",
    "goal_missing": "targets what is still missing for the goal",
}


@dataclass
class Evidence:
    """The rows behind a recommendation."""

    alumni_count: int
    similar_alumni_count: int
    # Whether these numbers describe the exact role and industry, or the role
    # archetype across every industry because the exact pair was too rare.
    basis: str = "this role and industry"
    example_campus_ids: list[str] = field(default_factory=list)
    median_first_salary_usd: int | None = None
    top_job_families: list[tuple[str, float]] = field(default_factory=list)
    top_industries: list[tuple[str, float]] = field(default_factory=list)
    goal_family_share: float | None = None


@dataclass
class Step:
    action_type: str
    position: str
    industry: str | None
    target_term: str
    # Probability the student's next step is this kind of opportunity, summed over
    # industries where the opportunity has them. The trustworthy half of the call.
    feasibility: float
    # P(this industry | this archetype). Reported separately and never folded into
    # `feasibility`, because industry is close to unpredictable in this data.
    industry_confidence: float | None
    goal_alignment: float
    goal_uplift: float
    projected_salary: tuple[int, int, int]
    prob_reaches_goal_family: float
    prob_reaches_goal_industry: float | None
    skills_added: list[str]
    skills_still_missing: list[str]
    why: list[str]
    evidence: Evidence
    action_id: str
    score: float
    next_steps: list["Step"] = field(default_factory=list)


# Below this many alumni, the evidence for one exact role-and-industry pair is
# too thin to say anything, and it is reported for the role archetype instead.
MIN_EVIDENCE = 10


@lru_cache(maxsize=1)
def _takers() -> dict[str, list[str]]:
    """Which alumni took each action, for the evidence layer."""
    index: dict[str, list[str]] = {}
    for campus_id, timeline in datasets.timelines().items():
        if not timeline.is_alumnus:
            continue
        for term_courses in timeline.courses.values():
            for course_id, *_ in term_courses:
                index.setdefault(f"{action_space.COURSE}|{course_id}", []).append(
                    campus_id
                )
        for term_experiences in timeline.experiences.values():
            for experience_type, name, industry, *_ in term_experiences:
                label = "-" if industry == data.NOT_APPLICABLE else industry
                index.setdefault(f"{experience_type}|{name}|{label}", []).append(
                    campus_id
                )
        for _, family, seniority, industry, _, _ in timeline.spells:
            index.setdefault(
                f"{action_space.ROLE}|{family}|{seniority}|{industry}", []
            ).append(campus_id)
    return index


@lru_cache(maxsize=1)
def _takers_by_position() -> dict[str, list[str]]:
    """Alumni grouped by role archetype, ignoring which industry they did it in."""
    index: dict[str, list[str]] = {}
    catalog = action_space.by_id()
    for action_id, campus_ids in _takers().items():
        action = catalog.get(action_id)
        if action is None:
            continue
        index.setdefault(action.position, []).extend(campus_ids)
    return index


def _phrase(feature: str) -> str:
    if feature in FEATURE_PHRASES:
        return FEATURE_PHRASES[feature]
    if feature.startswith("intern_affinity_"):
        return "the student's internships already point this way"
    if feature.startswith("affinity_"):
        return "the student's activities already point this way"
    if feature.startswith("fit_"):
        return "coursework already lines up with this job family"
    if feature.startswith("touched_"):
        return "the student has already worked in this industry"
    if feature.startswith("n_"):
        return "fits the student's record of activities"
    return feature.replace("_", " ")


def _kind_key(action: action_space.Action) -> str:
    if action.is_course:
        return "Course"
    if action.action_type in skills.INTERNSHIP_TYPES:
        return "Internship"
    return action.action_type


@dataclass
class _Shortlist:
    actions: list[action_space.Action]
    feasibility: np.ndarray
    # Probability mass on each role archetype, summed over its industry variants.
    # This, not the score of a single variant, is what "how likely is this student
    # to do a cybersecurity internship next" means, and it is the part of the
    # prediction that holds up. Keyed by position name.
    position_total: dict[str, float]
    # P(this industry | this archetype), from the feasibility mass over the full
    # candidate set rather than the shortlist, so it is a real conditional.
    industry_share: dict[str, float]
    # Shortlist positions that are industry variants of one archetype, grouped so
    # only the best of them survives into the final ranking.
    variant_groups: dict[int, str] = field(default_factory=dict)


class Planner:
    def __init__(
        self,
        next_step: NextStepModel | None = None,
        outcome: OutcomeModel | None = None,
        explain: bool = True,
    ) -> None:
        self.next_step = next_step or NextStepModel.load()
        self.outcome = outcome or OutcomeModel.load()
        self.explain = explain

    # -- candidate generation --------------------------------------------

    def _shortlist(
        self, state: state_module.State, goal: state_module.Goal
    ) -> _Shortlist:
        candidates = action_space.candidates(state)
        if not candidates:
            return _Shortlist([], np.zeros(0), {}, {})

        feasibility = self.next_step.score_actions(state, candidates, goal)

        position_total: dict[str, float] = {}
        for action, score in zip(candidates, feasibility):
            position_total[action.position] = position_total.get(
                action.position, 0.0
            ) + float(score)
        position_total = {name: min(total, 1.0) for name, total in position_total.items()}
        industry_share = {
            action.action_id: (
                float(score) / position_total[action.position]
                if position_total[action.position] > 0
                else 0.0
            )
            for action, score in zip(candidates, feasibility)
        }

        keep: list[int] = []
        variant_groups: dict[int, str] = {}
        by_kind: dict[str, list[int]] = {}
        for index, action in enumerate(candidates):
            key = _kind_key(action)
            by_kind.setdefault(key if key in QUOTAS else "other", []).append(index)

        for kind, indices in by_kind.items():
            quota = QUOTAS[kind]
            indices.sort(key=lambda index: -feasibility[index])

            if kind != "Internship":
                keep.extend(indices[:quota])
                continue

            # Keep the strongest archetypes, and several industries within each,
            # so the choice of industry is settled by goal uplift further down.
            grouped: dict[str, list[int]] = {}
            for index in indices:
                grouped.setdefault(candidates[index].position, []).append(index)
            best_archetypes = sorted(
                grouped, key=lambda name: -position_total.get(name, 0.0)
            )[:quota]
            for name in best_archetypes:
                variants = grouped[name][:INDUSTRY_VARIANTS]
                # Always cost out the industry the student actually asked for, even
                # if it is not among the most likely ones for this archetype.
                if goal.industry is not None:
                    preferred = [
                        index
                        for index in grouped[name]
                        if candidates[index].industry == goal.industry
                    ]
                    variants = list(dict.fromkeys(variants + preferred))
                for index in variants:
                    keep.append(index)
                    variant_groups[len(keep) - 1] = name

        return _Shortlist(
            actions=[candidates[index] for index in keep],
            feasibility=feasibility[keep],
            position_total=position_total,
            industry_share=industry_share,
            variant_groups=variant_groups,
        )

    # -- counterfactual scoring ------------------------------------------

    def _evaluate(
        self,
        state: state_module.State,
        goal: state_module.Goal,
        shortlist: _Shortlist,
    ) -> tuple[pd.DataFrame, list[state_module.State]]:
        successors = [
            state_module.transition(state, action) for action in shortlist.actions
        ]
        frame = datasets.to_frame(
            [state_module.state_features(item) for item in successors]
        )
        alignment = self.outcome.goal_score(frame, goal)
        families = self.outcome.family_distribution(frame)
        industries = self.outcome.industry_distribution(frame)
        quantiles = self.outcome.salary_quantiles(frame)

        # Absolute probability mass on the archetype, summed over its industries.
        # Reported to the student, but deliberately not used raw for ranking.
        effective = np.array(
            [
                shortlist.position_total.get(action.position, float(score))
                for action, score in zip(shortlist.actions, shortlist.feasibility)
            ]
        )

        baseline = self.outcome.score_state(state, goal)
        uplift = alignment - baseline

        # Both terms are rescaled before being combined, for different reasons.
        #
        # Feasibility is normalised *within each kind of step*. Absolutely, a
        # student takes a course almost every term and starts an internship almost
        # never, so P(course) is near 1 while P(a given internship) is near 0.06 --
        # rank on that and coursework wins every slot forever. The question worth
        # asking is not "what will happen next" but "which internship is a
        # realistic one for this student", so feasibility is measured against the
        # best option of the same kind and then uplift decides between kinds.
        #
        # Uplift is rescaled against the best candidate at this node because one
        # term moves the outcome distribution by only thousandths; on its raw scale
        # it would vanish next to the feasibility floor.
        relative_feasibility = np.ones(len(shortlist.actions))
        for kind in {_kind_key(action) for action in shortlist.actions}:
            mask = np.array(
                [_kind_key(action) == kind for action in shortlist.actions]
            )
            peak_feasibility = float(effective[mask].max())
            if peak_feasibility > 1e-9:
                relative_feasibility[mask] = effective[mask] / peak_feasibility

        positive = np.clip(uplift, 0.0, None)
        peak = float(positive.max())
        relative_uplift = positive / peak if peak > 1e-9 else positive
        score = relative_feasibility * (FEASIBILITY_BONUS + relative_uplift)

        scores = pd.DataFrame(
            {
                "feasibility": effective,
                "variant_feasibility": shortlist.feasibility,
                "alignment": alignment,
                "uplift": uplift,
                "score": score,
                "p10": quantiles[0.1].to_numpy(),
                "p50": quantiles[0.5].to_numpy(),
                "p90": quantiles[0.9].to_numpy(),
                "prob_family": (
                    families[goal.job_family].to_numpy()
                    if goal.job_family in families.columns
                    else np.zeros(len(shortlist.actions))
                ),
                "prob_industry": (
                    industries[goal.industry].to_numpy()
                    if goal.industry and goal.industry in industries.columns
                    else np.full(len(shortlist.actions), np.nan)
                ),
                "industry_share": [
                    shortlist.industry_share.get(action.action_id, np.nan)
                    if action.industry
                    else np.nan
                    for action in shortlist.actions
                ],
            }
        )
        return scores, successors

    def _diversify(
        self, ranked: list[tuple[float, int, action_space.Action]], width: int
    ) -> list[tuple[float, int, action_space.Action]]:
        """Greedily pick high scorers while discouraging repeats of a kind.

        Without this the top three are usually three courses, which is accurate
        and useless: the student wants to see the different kinds of move
        available to them.
        """
        chosen: list[tuple[float, int, action_space.Action]] = []
        seen: dict[str, int] = {}
        pool = list(ranked)
        while pool and len(chosen) < width:
            best_position, best_value = 0, -np.inf
            for position, (score, _, action) in enumerate(pool):
                adjusted = score * (DIVERSITY_PENALTY ** seen.get(_kind_key(action), 0))
                if adjusted > best_value:
                    best_position, best_value = position, adjusted
            entry = pool.pop(best_position)
            key = _kind_key(entry[2])
            seen[key] = seen.get(key, 0) + 1
            chosen.append(entry)
        return chosen

    # -- evidence --------------------------------------------------------

    def evidence_for(
        self,
        action: action_space.Action,
        state: state_module.State,
        goal: state_module.Goal,
    ) -> Evidence:
        """Who actually did this, and what happened to them.

        Restricted to alumni in the same major where possible, and to those whose
        outcome is known at all -- the 480 graduates who did not respond are not
        evidence of anything.
        """
        outcomes = datasets.outcomes()
        timelines = datasets.timelines()

        def resolve(ids: list[str]) -> list[str]:
            # A person appears once per term they did something, so deduplicate
            # before counting or the evidence overstates how many people this is.
            return list(
                dict.fromkeys(
                    campus_id for campus_id in ids if campus_id in outcomes
                )
            )

        def cohort(ids: list[str]) -> tuple[list[str], list[str]]:
            known = resolve(ids)
            same_major = [
                campus_id
                for campus_id in known
                if timelines[campus_id].major == state.major
            ]
            return known, same_major or known

        # Widen only after the same-major filter has been applied. Testing the
        # count before it reports five people as evidence about one industry when
        # the same role across industries has fifty.
        known, similar = cohort(_takers().get(action.action_id, []))
        basis = "this role and industry"
        if len(similar) < MIN_EVIDENCE:
            wider_known, wider_similar = cohort(
                _takers_by_position().get(action.position, [])
            )
            if len(wider_similar) > len(similar):
                known, similar = wider_known, wider_similar
                basis = "this role across all industries"

        if not similar:
            return Evidence(
                alumni_count=len(known), similar_alumni_count=0, basis=basis
            )

        records = [outcomes[campus_id] for campus_id in similar]
        salaries = [
            record["salary_real"]
            for record in records
            if record["salary_real"] is not None
        ]
        families = pd.Series([record["outcome_family"] for record in records])
        industries = pd.Series(
            [
                record["outcome_industry"]
                for record in records
                if record["outcome_industry"] != datasets.NOT_EMPLOYED_INDUSTRY
            ]
        )
        family_shares = families.value_counts(normalize=True)

        return Evidence(
            alumni_count=len(known),
            similar_alumni_count=len(similar),
            basis=basis,
            example_campus_ids=similar[:5],
            median_first_salary_usd=(
                int(np.median(salaries)) if salaries else None
            ),
            top_job_families=[
                (str(name), round(float(share), 3))
                for name, share in family_shares.head(3).items()
            ],
            top_industries=[
                (str(name), round(float(share), 3))
                for name, share in industries.value_counts(normalize=True)
                .head(3)
                .items()
            ],
            goal_family_share=(
                round(float(family_shares.get(goal.job_family, 0.0)), 3)
            ),
        )

    # -- search ----------------------------------------------------------

    def recommend(
        self,
        state: state_module.State,
        goal: state_module.Goal,
        width: int = BEAM_WIDTH,
        depth: int = BEAM_DEPTH,
    ) -> list[Step]:
        return self._expand(state, goal, width, depth, top_level=True)

    def rank(
        self,
        state: state_module.State,
        goal: state_module.Goal,
        width: int = BEAM_WIDTH,
    ) -> tuple[list[tuple[float, int, action_space.Action]], pd.DataFrame, list]:
        """The chosen steps at one node, without building the report around them.

        Shared by the recommender and the backtest, so the evaluation scores the
        same ranking the API returns rather than a reimplementation of it.
        """
        shortlist = self._shortlist(state, goal)
        if not shortlist.actions:
            return [], pd.DataFrame(), []
        scores, successors = self._evaluate(state, goal, shortlist)

        # Collapse each archetype's industry variants to a single representative, so
        # the shortlist competes on distinct opportunities. A stated target industry
        # wins outright: the industry head is weak enough that overriding what the
        # student told us with it would be guessing dressed up as a prediction.
        def variant_rank(position: int) -> tuple[int, float]:
            action = shortlist.actions[position]
            preferred = goal.industry is not None and action.industry == goal.industry
            return (1 if preferred else 0, float(scores["score"].iloc[position]))

        best_variant: dict[str, int] = {}
        for position, group in shortlist.variant_groups.items():
            incumbent = best_variant.get(group)
            if incumbent is None or variant_rank(position) > variant_rank(incumbent):
                best_variant[group] = position
        survivors = set(best_variant.values())

        ranked = sorted(
            (
                (float(scores["score"].iloc[position]), position, action)
                for position, action in enumerate(shortlist.actions)
                if position not in shortlist.variant_groups or position in survivors
            ),
            key=lambda item: -item[0],
        )
        return self._diversify(ranked, width), scores, successors

    def top_action_ids(
        self,
        state: state_module.State,
        goal: state_module.Goal,
        width: int = BEAM_WIDTH,
    ) -> list[str]:
        chosen, _, _ = self.rank(state, goal, width)
        return [action.action_id for _, _, action in chosen]

    def _expand(
        self,
        state: state_module.State,
        goal: state_module.Goal,
        width: int,
        depth: int,
        top_level: bool = False,
    ) -> list[Step]:
        if depth <= 0:
            return []
        chosen, scores, successors = self.rank(state, goal, width)
        if not chosen:
            return []

        # SHAP is only worth its cost on the steps the student will actually read.
        explanations = (
            self.next_step.explain(state, [action for _, _, action in chosen], goal)
            if self.explain and top_level
            else [[] for _ in chosen]
        )

        goal_profile = goal.skill_profile
        steps: list[Step] = []
        for (score, position, action), contributions in zip(chosen, explanations):
            successor = successors[position]
            row = scores.iloc[position]
            gained = np.clip(action.skill_profile - state.skill_vector, 0.0, None)
            steps.append(
                Step(
                    action_type=action.action_type,
                    position=action.position,
                    industry=action.industry,
                    target_term=terms.term_label(state.term_index),
                    feasibility=float(row["feasibility"]),
                    industry_confidence=(
                        None
                        if action.industry is None or np.isnan(row["industry_share"])
                        else round(float(row["industry_share"]), 4)
                    ),
                    goal_alignment=float(row["alignment"]),
                    goal_uplift=float(row["uplift"]),
                    projected_salary=(
                        int(row["p10"]),
                        int(row["p50"]),
                        int(row["p90"]),
                    ),
                    prob_reaches_goal_family=float(row["prob_family"]),
                    prob_reaches_goal_industry=(
                        None
                        if np.isnan(row["prob_industry"])
                        else float(row["prob_industry"])
                    ),
                    skills_added=skills.names(gained, 0.3),
                    skills_still_missing=skills.missing(
                        successor.skill_vector, goal_profile
                    ),
                    why=[_phrase(name) for name, _ in contributions],
                    evidence=self.evidence_for(action, state, goal),
                    action_id=action.action_id,
                    score=float(score),
                    # Deeper levels narrow: the point is to show where a choice
                    # leads, not to enumerate an exponential tree.
                    next_steps=self._expand(
                        successor, goal, max(2, width - 1), depth - 1
                    ),
                )
            )
        return steps


def describe(steps: list[Step], indent: int = 0) -> str:
    """A plain-text rendering, used by the CLI and for eyeballing output."""
    lines = []
    for step in steps:
        pad = "  " * indent
        where = f" in {step.industry}" if step.industry else ""
        low, median, high = step.projected_salary
        lines.append(
            f"{pad}- [{step.action_type}] {step.position}{where}  ({step.target_term})"
        )
        lines.append(
            f"{pad}  feasibility {step.feasibility:.2f}"
            + (
                f" | industry {step.industry_confidence:.2f}"
                if step.industry_confidence is not None
                else ""
            )
            + f" | goal alignment {step.goal_alignment:.3f} "
            f"(uplift {step.goal_uplift:+.3f})"
        )
        lines.append(
            f"{pad}  projected first salary ${low:,}-${high:,} (median ${median:,})"
            f" | P(goal family) {step.prob_reaches_goal_family:.2f}"
        )
        if step.skills_added:
            lines.append(f"{pad}  adds: {', '.join(step.skills_added)}")
        if step.skills_still_missing:
            lines.append(
                f"{pad}  still missing: {', '.join(step.skills_still_missing[:6])}"
            )
        if step.why:
            lines.append(f"{pad}  why: {'; '.join(step.why)}")
        evidence = step.evidence
        if evidence.similar_alumni_count:
            families = ", ".join(
                f"{name} {share:.0%}" for name, share in evidence.top_job_families
            )
            salary = (
                f", median first salary ${evidence.median_first_salary_usd:,}"
                if evidence.median_first_salary_usd
                else ""
            )
            lines.append(
                f"{pad}  evidence ({evidence.basis}): "
                f"{evidence.similar_alumni_count} same-major alumni did this{salary}; "
                f"they went to {families}"
            )
            lines.append(f"{pad}  e.g. {', '.join(evidence.example_campus_ids)}")
        if step.next_steps:
            lines.append(f"{pad}  then:")
            lines.append(describe(step.next_steps, indent + 2))
    return "\n".join(lines)
