"""Academic terms as a totally ordered integer index.

Terms are written `Fall 2023`. Within a year the order is Spring, Summer, Fall,
so a term maps to `year * 3 + season_offset` and ordinary integer comparison
gives chronological order.
"""

from __future__ import annotations

import re
from functools import lru_cache

SEASONS = ("Spring", "Summer", "Fall")
SEASON_OFFSET = {season: offset for offset, season in enumerate(SEASONS)}

# Month a term begins, used to place a job start_date into a term.
SEASON_START_MONTH = {"Spring": 1, "Summer": 6, "Fall": 8}

_TERM_PATTERN = re.compile(r"^\s*(Spring|Summer|Fall)\s+(\d{4})\s*$")


@lru_cache(maxsize=4096)
def term_index(term: str) -> int:
    match = _TERM_PATTERN.match(str(term))
    if match is None:
        raise ValueError(f"Unparseable term {term!r}")
    season, year = match.group(1), int(match.group(2))
    return year * 3 + SEASON_OFFSET[season]


def term_label(index: int) -> str:
    year, offset = divmod(int(index), 3)
    return f"{SEASONS[offset]} {year}"


def season_of(index: int) -> str:
    return SEASONS[int(index) % 3]


def year_of(index: int) -> int:
    return int(index) // 3


def next_term(index: int, skip_summer: bool = True) -> int:
    """The term after `index`. Summer is skipped by default: most enrollment is
    Fall and Spring, so a plan that assumes a Summer term would overstate pace.
    """
    step = index + 1
    if skip_summer and season_of(step) == "Summer":
        step += 1
    return step


def term_range(start: int, stop: int, skip_summer: bool = True) -> list[int]:
    terms: list[int] = []
    current = start
    while current < stop:
        if skip_summer and season_of(current) == "Summer":
            current += 1
            continue
        terms.append(current)
        current += 1
    return terms


def index_from_date(date: str) -> int:
    """Place an ISO date into the term that was underway when it started."""
    year, month, _ = (int(part) for part in str(date)[:10].split("-"))
    if month >= SEASON_START_MONTH["Fall"]:
        season = "Fall"
    elif month >= SEASON_START_MONTH["Summer"]:
        season = "Summer"
    else:
        season = "Spring"
    return year * 3 + SEASON_OFFSET[season]


def is_term(value: object) -> bool:
    return isinstance(value, str) and _TERM_PATTERN.match(value) is not None
