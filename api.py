"""Translation layer between what a student types and what the models consume.

The one thing here that is not mechanical is turning a free-text goal such as
"senior ML engineer at a software company" into the structured target the outcome
model scores against. That is resolved locally against the job titles and
industries that exist in the data, and only falls back to a Gemini embedding when
the local match is genuinely poor -- so the service keeps working with no network
and no API key, and the ranking never depends on a language model.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from model import data, skills
from model import state as state_module
from model.planner import Planner, Step

ENV_PATH = Path(__file__).resolve().parent / ".env"

# Below this token overlap the local match is not trustworthy and, if a key is
# configured, an embedding is worth the round trip.
LOCAL_MATCH_FLOOR = 0.34

SENIORITY_KEYWORDS = (
    ("director", "Director"),
    ("vp", "Director"),
    ("head of", "Director"),
    ("manager", "Manager"),
    ("lead", "Lead"),
    ("principal", "Lead"),
    ("staff", "Senior"),
    ("senior", "Senior"),
    ("sr.", "Senior"),
    ("mid", "Mid"),
    ("junior", "Entry"),
    ("entry", "Entry"),
    ("intern", "Entry"),
    ("associate", "Entry"),
    ("graduate", "Entry"),
)


def _load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env()


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9+]+", str(text).lower()) if token}


def _overlap(left: str, right: str) -> float:
    first, second = _tokens(left), _tokens(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


@lru_cache(maxsize=1)
def _title_families() -> dict[str, str]:
    """Every job title in the data, mapped to its job family."""
    employment = data.load("employment_history")
    return (
        employment.groupby("job_title")["job_family"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )


# Words that say how senior a role is, or nothing at all about its subject.
# Left in, they dominate the token overlap: "senior cybersecurity analyst" matches
# "Senior Systems Analyst" on two generic words and loses to it on the one word
# that carries the meaning.
GENERIC_TITLE_WORDS = frozenset(
    {level.lower() for level, _ in SENIORITY_KEYWORDS}
    | {
        "engineer",
        "analyst",
        "specialist",
        "developer",
        "scientist",
        "administrator",
        "consultant",
        "architect",
        "of",
        "the",
        "a",
        "at",
        "in",
        "for",
    }
)


def _score_match(query: str, option: str) -> float:
    """How well a free-text query names this option.

    Blends token overlap with containment, because the two fail in opposite
    cases: overlap misses "healthcare" against "Healthcare & Life Sciences",
    while containment misses "ML engineer" against "Machine Learning Engineer".
    """
    normalized = str(query).strip().lower()
    lowered = option.lower()
    if lowered == normalized:
        return 1.0

    score = _overlap(query, option)
    if normalized and (normalized in lowered or lowered in normalized):
        score = max(score, 0.7)

    # An option whose every meaningful word appears in the query is named by it,
    # however much else the query says. Scaled by how many distinctive words that
    # is, so "IT Support Specialist" outranks "IT Analyst" for "IT support
    # technician" instead of tying it on the single word they share.
    option_tokens = _tokens(option) - GENERIC_TITLE_WORDS
    if option_tokens and option_tokens <= _tokens(query):
        score = max(score, min(0.85, 0.55 + 0.15 * len(option_tokens)))
    return score


def _best_local(query: str, options: list[str]) -> tuple[str | None, float]:
    scored = [(option, _score_match(query, option)) for option in options]
    scored.sort(key=lambda item: -item[1])
    best, score = scored[0]
    return (best, score) if score > 0 else (None, 0.0)


def _subject_only(job_title: str) -> str:
    """The query with seniority and generic role words removed."""
    remaining = [
        token for token in _tokens(job_title) if token not in GENERIC_TITLE_WORDS
    ]
    return " ".join(remaining) if remaining else str(job_title)


def _embed(text: str) -> list[float] | None:
    """Unit-norm embedding of a free-text goal, or None if unavailable."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.embed_content(
            model="gemini-embedding-2", contents=text
        )
        if not response.embeddings or response.embeddings[0].values is None:
            return None
        vector = np.array(response.embeddings[0].values, dtype=float)
        return (vector / max(float(np.linalg.norm(vector)), 1e-12)).tolist()
    except Exception:
        # A goal that cannot be embedded is not a reason to fail the request; the
        # local match already produced a usable answer.
        return None


def _nearest_by_embedding(query: str, options: list[str]) -> str | None:
    import tiger_db

    vector = _embed(query)
    if vector is None:
        return None
    try:
        stored = tiger_db.load_scoring_data()[1]
    except Exception:
        return None
    candidates = {
        label: embedding for label, embedding in stored.items() if label in options
    }
    if not candidates:
        return None
    goal = np.asarray(vector, dtype=float)
    scored = {
        label: float(np.asarray(embedding, dtype=float) @ goal)
        for label, embedding in candidates.items()
    }
    return max(scored, key=lambda label: scored[label])


def infer_seniority(job_title: str) -> str:
    haystack = str(job_title).lower()
    for keyword, level in SENIORITY_KEYWORDS:
        if keyword in haystack:
            return level
    return "Entry"


def resolve_goal(
    job_title: str,
    industry: str | None = None,
    salary_usd: float | None = None,
) -> state_module.Goal:
    """Map free text onto a job family, industry and seniority that exist."""
    families = list(skills.job_families())
    titles = _title_families()

    # Try the query with seniority and generic role words stripped, and also as
    # written. Stripping is what makes "senior cybersecurity analyst" land in
    # Cybersecurity rather than matching "Senior Systems Analyst" on two words
    # that say nothing about the subject; keeping the original is what still
    # resolves "backend developer", where the distinctive word appears in no
    # title in this data and only the generic one can carry it.
    family, score = None, 0.0
    for query in (_subject_only(job_title), str(job_title)):
        for candidate, candidate_score in (
            _best_local(query, families),
            (lambda pair: (titles.get(pair[0]), pair[1]))(
                _best_local(query, list(titles))
            ),
        ):
            if candidate is not None and candidate_score > score:
                family, score = candidate, candidate_score

    if score < LOCAL_MATCH_FLOOR:
        nearest = _nearest_by_embedding(job_title, list(titles))
        if nearest is not None:
            family = titles[nearest]
        # With no embedding available a weak local match still beats refusing to
        # answer, so it stands; only a query that matched nothing at all fails.
    if family is None:
        raise ValueError(
            f"Could not map {job_title!r} onto a job family. Known families: "
            + ", ".join(families)
        )

    resolved_industry = None
    if industry:
        matched, industry_score = _best_local(industry, list(skills.industries()))
        resolved_industry = matched if industry_score >= LOCAL_MATCH_FLOOR else None

    return state_module.Goal(
        job_family=family,
        industry=resolved_industry,
        salary_usd=salary_usd,
        job_title=job_title,
        seniority=infer_seniority(job_title),
    )


class CareerAdvisor:
    """Loads the models once and answers plan requests."""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()

    def plan(
        self,
        major: str,
        track: str,
        gpa: float | None,
        credits_earned: float,
        classes: list[str],
        experiences: list[dict],
        job_title: str,
        industry: str | None = None,
        salary: float | None = None,
        current_term: str = "Fall 2026",
        entry_term: str | None = None,
        entry_type: str = "First-Time Freshman",
        work_hours: float = 0.0,
        campus_id: str | None = None,
        width: int = 3,
        depth: int = 3,
    ) -> tuple[list[Step], state_module.Goal, state_module.State]:
        goal = resolve_goal(job_title, industry, salary)
        student = state_module.from_request(
            major=major,
            track=track,
            gpa=gpa,
            credits_earned=credits_earned,
            completed_courses=classes,
            experiences=experiences,
            current_term=current_term,
            entry_term=entry_term,
            entry_type=entry_type,
            work_hours=work_hours,
            campus_id=campus_id,
        )
        steps = self.planner.recommend(student, goal, width=width, depth=depth)
        return steps, goal, student
