"""Analysis-ready experience table.

Every row is kept. The previous version filtered `role_level != "Not Applicable"`,
which looks harmless but deleted all 4,106 internships, 678 co-ops and 2,845
certifications, because none of those experience types carry a role level. Those
7,629 rows are the opportunities the recommender is built to suggest, so instead of
dropping them the sentinel is simply recognised for what it is.
"""

from __future__ import annotations

import paths

from model import data
from model.state import OUTCOME_QUALITY, ROLE_LEVEL_RANK
from model import skills


def add_kind_flags(experience):
    frame = experience.copy()
    frame["is_internship"] = frame["experience_type"].isin(skills.INTERNSHIP_TYPES)
    frame["is_credential"] = frame["experience_type"] == "Certification"
    frame["has_role_level"] = frame["role_level"] != data.NOT_APPLICABLE
    return frame


def add_quality(experience):
    """Score how well each activity went, on one scale across all the types.

    `outcome` uses a different vocabulary per experience type -- a hackathon is
    won, an internship converts to a return offer, a certification is earned -- so
    comparing involvement across types needs them mapped onto something common.
    """
    frame = experience.copy()
    frame["outcome_quality"] = frame["outcome"].map(OUTCOME_QUALITY)
    frame["role_level_rank"] = frame["role_level"].map(ROLE_LEVEL_RANK).fillna(0)
    return frame


def add_family_affinity(experience):
    """Which job family each activity empirically points toward, and how strongly."""
    frame = experience.copy()
    affinity = [skills.activity_affinity(name) for name in frame["experience_name"]]
    frame["points_to_family"] = [family for family, _ in affinity]
    frame["family_lift"] = [round(lift, 4) for _, lift in affinity]
    return frame


def main() -> None:
    experience = data.load("student_experience")
    experience = add_kind_flags(experience)
    experience = add_quality(experience)
    experience = add_family_affinity(experience)
    paths.write(experience, "student_experience_clean")

    print("\nevery experience type is kept:")
    print(experience["experience_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
