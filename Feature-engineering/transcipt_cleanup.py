"""Analysis-ready transcript table.

`grade_points` and `credits_earned` are kept. The previous version dropped both,
which makes it impossible to recompute a GPA as of a given term -- and an as-of-term
GPA is the whole point, since the stored `final_gpa` on the alumni table describes
someone at graduation and cannot be fed to a model reasoning about a sophomore.
"""

from __future__ import annotations

import paths

from model import data, skills, terms


def add_term_index(transcripts):
    """A sortable integer per term, so a transcript can be walked in order."""
    frame = transcripts.copy()
    frame["term_index"] = frame["term"].map(terms.term_index)
    return frame


def add_grade_flags(transcripts):
    frame = transcripts.copy()
    # W and IP carry no grade points and are excluded from GPA; F earns no credit
    # but does count against it.
    frame["counts_toward_gpa"] = ~frame["grade"].isin(["W", "IP"])
    frame["passed"] = ~frame["grade"].isin(["W", "IP", "F"])
    return frame


def add_skill_count(transcripts):
    """How many catalog skills each attempt covers, for skill-gap work."""
    frame = transcripts.copy()
    profiles = skills.course_profiles()
    frame["skill_count"] = [
        int((profiles[course_id] >= 0.5).sum()) if course_id in profiles else 0
        for course_id in frame["course_id"]
    ]
    return frame


def main() -> None:
    transcripts = data.load("transcripts")
    transcripts = add_term_index(transcripts)
    transcripts = add_grade_flags(transcripts)
    transcripts = add_skill_count(transcripts)
    paths.write(transcripts, "transcripts_clean")

    print("\ngrade distribution, all attempts kept:")
    print(transcripts["grade"].value_counts().to_string())


if __name__ == "__main__":
    main()
