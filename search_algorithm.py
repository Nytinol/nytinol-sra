import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "Feature-engineering"))

import tiger_db


TITLE_WEIGHT = 0.50
INDUSTRY_WEIGHT = 0.30
SALARY_WEIGHT = 0.20
GRADE_WEIGHT = {"A": 1.0, "B": 0.75, "C": 0.4, "D": 0.15, "F": 0.0}
EXPERIENCE_WEIGHT = {
    "Internship": 1.0,
    "Co-op": 0.95,
    "Undergraduate Research": 0.85,
    "Certification": 0.7,
    "Hackathon": 0.65,
    "Student Organization": 0.6,
    "Competitive Team": 0.6,
    "Campus Job": 0.55,
    "Tutoring": 0.5,
    "Peer Mentor": 0.5,
}
EXPERIENCE_BUCKET = {
    "Internship": "internship",
    "Co-op": "internship",
    "Internships": "internship",
    "Undergraduate Research": "research",
    "Research": "research",
    "Student Organization": "club",
    "Competitive Team": "club",
    "Peer Mentor": "club",
    "Club": "club",
    "Clubs": "club",
    "Hackathon": "hackathon",
    "Hackathons": "hackathon",
    "Certification": "certification",
    "Certifications": "certification",
    "Campus Job": "campus_job",
    "Campus Jobs": "campus_job",
    "Tutoring": "campus_job",
}


@dataclass
class Achievement:
    experience_type: str
    experience_name: str
    industry: str
    term: str
    terms_participated: int
    hours_per_week: float | None = None


@dataclass
class CurrentAchievements:
    name: str
    age: int
    major: str
    experiences: list[Achievement] = field(default_factory=list)


def _salary_score(alumni_salary: float, goal_salary: float) -> float:
    if goal_salary <= 0:
        return 0.0
    gap = abs(alumni_salary - goal_salary) / goal_salary
    return max(0.0, 1.0 - gap)


def _goal_score(alum: dict, title_score: float, industry_score: float, salary: float) -> float:
    pay_score = _salary_score(alum["first_job_annual_salary_usd"], salary)
    return (
        TITLE_WEIGHT * title_score
        + INDUSTRY_WEIGHT * industry_score
        + SALARY_WEIGHT * pay_score
    )


def _completed(classes: list[str]) -> set[str]:
    return {item.strip().lower() for item in classes}


def _bucket(experience_type: str) -> str | None:
    return EXPERIENCE_BUCKET.get(experience_type)


def _investment(achievements: CurrentAchievements) -> dict[str, float]:
    totals = {
        "internship": 0.0,
        "club": 0.0,
        "research": 0.0,
        "hackathon": 0.0,
        "certification": 0.0,
        "campus_job": 0.0,
    }
    for experience in achievements.experiences:
        bucket = _bucket(experience.experience_type)
        if bucket is None:
            continue
        terms = max(experience.terms_participated, 1)
        if experience.hours_per_week:
            totals[bucket] += terms * experience.hours_per_week
        else:
            totals[bucket] += terms * 8
    return totals


def _gap_multiplier(bucket: str | None, investment: dict[str, float], age: int) -> float:
    if bucket is None:
        return 1.0
    total = sum(investment.values())
    share = 0.0 if total <= 0 else investment[bucket] / total
    # A bucket the student has not touched stays attractive. A bucket that
    # already dominates their time is scaled down so the other kinds can surface.
    multiplier = 1.75 if total <= 0 else 0.35 + (1 - share) * 1.4
    if age >= 22 and bucket == "club":
        multiplier *= 0.8
    if age >= 22 and bucket in {"internship", "research", "certification"}:
        multiplier *= 1.1
    return multiplier


def _next_steps(
    alumni: list[dict],
    records: dict[str, list[dict]],
    classes: list[str],
    major: str,
    gpa: float,
    achievements: CurrentAchievements,
) -> list[str]:
    completed = _completed(classes)
    weights = {alum["campus_id"]: alum["score"] for alum in alumni}
    catalog = {row["course_id"]: row for row in records["course_catalog"]}
    course_scores: dict[str, float] = {}
    course_grades: dict[str, list[str]] = {}
    for row in records["transcripts"]:
        course_id = row["course_id"]
        title = str(row["course_title"])
        if course_id.lower() in completed or title.lower() in completed:
            continue
        grade = str(row["grade"])
        weight = weights.get(row["campus_id"], 0.0) * GRADE_WEIGHT.get(grade, 0.25)
        required = str(catalog.get(course_id, {}).get("required_for_majors", ""))
        if major and major in required:
            weight *= 1.25
        course_scores[course_id] = course_scores.get(course_id, 0.0) + weight
        course_grades.setdefault(course_id, []).append(grade)

    owned_names = {item.experience_name.strip().lower() for item in achievements.experiences}
    investment = _investment(achievements)
    experience_scores: dict[tuple[str, str], float] = {}
    experience_detail: dict[tuple[str, str], dict] = {}
    for row in records["student_experience"]:
        kind = str(row["experience_type"])
        if kind not in EXPERIENCE_WEIGHT:
            continue
        name = str(row["experience_name"])
        if name.strip().lower() in owned_names:
            continue
        key = (kind, name)
        bucket = _bucket(kind)
        experience_scores[key] = experience_scores.get(key, 0.0) + (
            weights.get(row["campus_id"], 0.0)
            * EXPERIENCE_WEIGHT[kind]
            * _gap_multiplier(bucket, investment, achievements.age)
        )
        experience_detail[key] = row

    candidates: list[tuple[float, str]] = []
    if course_scores:
        course_id = max(course_scores, key=lambda item: course_scores[item])
        title = catalog.get(course_id, {}).get("course_title", course_id)
        grades = ", ".join(course_grades[course_id])
        candidates.append(
            (
                course_scores[course_id],
                f"Take {course_id} ({title}) this year. The closest alumni earned {grades}, "
                f"and it is not in the classes already completed.",
            )
        )
    for key in sorted(experience_scores, key=lambda item: experience_scores[item], reverse=True)[:2]:
        detail = experience_detail[key]
        hours = detail.get("hours_per_week")
        if hours is None:
            hours = "Not Applicable"
        candidates.append(
            (
                experience_scores[key],
                "\n".join(
                    [
                        str(detail["experience_type"]),
                        str(detail["experience_name"]),
                        str(detail.get("industry") or "Not Applicable"),
                        str(detail.get("term")),
                        str(detail.get("duration_terms")),
                        str(hours),
                    ]
                ),
            )
        )
    if len(candidates) < 3:
        names = ", ".join(alum["campus_id"] for alum in alumni)
        average_gpa = sum(alum["final_gpa"] for alum in alumni) / len(alumni)
        candidates.append(
            (
                0.0,
                f"Raise GPA from {gpa:.2f} toward {average_gpa:.2f}, the average of {names}, "
                f"whose first jobs are the closest match to this goal.",
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in candidates[:3]]


def search_algorithm(
    job_title: list[float],
    current_achievments: CurrentAchievements,
    gpa: float,
    classes: list[str],
    major: str,
    salary: float,
    industry: list[float],
) -> list[str]:
    student_major = current_achievments.major or major
    alumni, title_vectors, industry_vectors = tiger_db.load_scoring_data()
    title_scores = {
        name: cosine_similarity(job_title, vector) for name, vector in title_vectors.items()
    }
    industry_scores = {
        name: cosine_similarity(industry, vector) for name, vector in industry_vectors.items()
    }
    ranked = []
    for alum in alumni:
        alum["score"] = _goal_score(
            alum,
            title_scores[alum["first_job_title"]],
            industry_scores[alum["first_employer_industry"]],
            salary,
        )
        ranked.append(alum)
    ranked.sort(key=lambda alum: alum["score"], reverse=True)
    closest = ranked[:3]
    records = tiger_db.load_campus_records([alum["campus_id"] for alum in closest])
    return _next_steps(closest, records, classes, student_major, gpa, current_achievments)


def cosine_similarity(goal_vector: list[float], job_vector: list[float]) -> float:
    dot = sum(left * right for left, right in zip(goal_vector, job_vector))
    goal_norm = sum(value * value for value in goal_vector) ** 0.5
    job_norm = sum(value * value for value in job_vector) ** 0.5
    return dot / (goal_norm * job_norm)

def closestJobTitles(goal_vector: list[float]) -> list[str]:
    job_vectors = tiger_db.get_all_job_vectors()
    similarities = [cosine_similarity(goal_vector, job_vector) for job_vector in job_vectors]
    return [job_vectors[i] for i in np.argsort(similarities)[::-1]]