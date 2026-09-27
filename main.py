"""HTTP surface for the career planner.

`POST /plan` returns structured, recursive recommendations rather than sentences,
so a client can render the branching roadmap and show the confidence and the
supporting alumni rows next to each step.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import CareerAdvisor
from model import skills
from model.planner import Evidence, Step

app = FastAPI(title="Career Pathways Planner")
advisor = CareerAdvisor()


class ExperienceIn(BaseModel):
    experience_type: str
    experience_name: str
    industry: str | None = None
    term: str | None = None
    role_level: str = "Member"
    outcome: str = "Completed"
    is_paid: bool | None = None


class PlanRequest(BaseModel):
    major: str
    track: str = "General"
    gpa: float | None = None
    credits_earned: float = 0
    classes: list[str] = Field(default_factory=list)
    experiences: list[ExperienceIn] = Field(default_factory=list)
    job_title: str
    industry: str | None = None
    salary: float | None = None
    current_term: str = "Fall 2026"
    entry_term: str | None = None
    entry_type: str = "First-Time Freshman"
    work_hours: float = 0
    campus_id: str | None = None
    width: int = Field(default=3, ge=1, le=5)
    depth: int = Field(default=3, ge=1, le=4)


class EvidenceOut(BaseModel):
    alumni_count: int
    similar_alumni_count: int
    basis: str
    example_campus_ids: list[str]
    median_first_salary_usd: int | None
    top_job_families: list[tuple[str, float]]
    top_industries: list[tuple[str, float]]
    goal_family_share: float | None


class SalaryBand(BaseModel):
    p10: int
    median: int
    p90: int


class RecommendationOut(BaseModel):
    action_type: str
    position: str
    industry: str | None
    target_term: str
    feasibility: float = Field(
        description=(
            "Probability the student's next step is this kind of opportunity, "
            "summed over industries. The reliable half of the prediction."
        )
    )
    industry_confidence: float | None = Field(
        default=None,
        description=(
            "P(this industry | this role). Reported separately because industry "
            "is close to unpredictable from an academic record in this data."
        ),
    )
    goal_alignment: float
    goal_uplift: float
    projected_salary: SalaryBand
    prob_reaches_goal_family: float
    prob_reaches_goal_industry: float | None
    skills_added: list[str]
    skills_still_missing: list[str]
    why: list[str]
    evidence: EvidenceOut
    next_steps: list["RecommendationOut"] = Field(default_factory=list)


RecommendationOut.model_rebuild()


class GoalOut(BaseModel):
    job_family: str
    industry: str | None
    seniority: str
    salary_usd: float | None
    requested_job_title: str | None


class StudentOut(BaseModel):
    class_level: str
    credits_earned: float
    gpa: float | None
    skills_held: list[str]
    internships_completed: int


class PlanResponse(BaseModel):
    goal: GoalOut
    student: StudentOut
    baseline_prob_goal_family: float
    recommendations: list[RecommendationOut]


def _evidence_out(evidence: Evidence) -> EvidenceOut:
    return EvidenceOut(
        alumni_count=evidence.alumni_count,
        similar_alumni_count=evidence.similar_alumni_count,
        basis=evidence.basis,
        example_campus_ids=evidence.example_campus_ids,
        median_first_salary_usd=evidence.median_first_salary_usd,
        top_job_families=evidence.top_job_families,
        top_industries=evidence.top_industries,
        goal_family_share=evidence.goal_family_share,
    )


def _recommendation_out(step: Step) -> RecommendationOut:
    low, median, high = step.projected_salary
    return RecommendationOut(
        action_type=step.action_type,
        position=step.position,
        industry=step.industry,
        target_term=step.target_term,
        feasibility=round(step.feasibility, 4),
        industry_confidence=step.industry_confidence,
        goal_alignment=round(step.goal_alignment, 4),
        goal_uplift=round(step.goal_uplift, 4),
        projected_salary=SalaryBand(p10=low, median=median, p90=high),
        prob_reaches_goal_family=round(step.prob_reaches_goal_family, 4),
        prob_reaches_goal_industry=(
            None
            if step.prob_reaches_goal_industry is None
            else round(step.prob_reaches_goal_industry, 4)
        ),
        skills_added=step.skills_added,
        skills_still_missing=step.skills_still_missing,
        why=step.why,
        evidence=_evidence_out(step.evidence),
        next_steps=[_recommendation_out(child) for child in step.next_steps],
    )


@app.post("/plan", response_model=PlanResponse)
def plan(body: PlanRequest) -> PlanResponse:
    try:
        steps, goal, student = advisor.plan(
            major=body.major,
            track=body.track,
            gpa=body.gpa,
            credits_earned=body.credits_earned,
            classes=body.classes,
            experiences=[item.model_dump() for item in body.experiences],
            job_title=body.job_title,
            industry=body.industry,
            salary=body.salary,
            current_term=body.current_term,
            entry_term=body.entry_term,
            entry_type=body.entry_type,
            work_hours=body.work_hours,
            campus_id=body.campus_id,
            width=body.width,
            depth=body.depth,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PlanResponse(
        goal=GoalOut(
            job_family=goal.job_family,
            industry=goal.industry,
            seniority=goal.seniority,
            salary_usd=goal.salary_usd,
            requested_job_title=goal.job_title,
        ),
        student=StudentOut(
            class_level=student.class_level,
            credits_earned=student.credits_earned,
            gpa=student.gpa,
            skills_held=skills.names(student.skill_vector),
            internships_completed=student.internship_count,
        ),
        baseline_prob_goal_family=round(
            advisor.planner.outcome.score_state(student, goal), 4
        ),
        recommendations=[_recommendation_out(step) for step in steps],
    )


@app.get("/reference")
def reference() -> dict:
    """What the planner knows how to aim at, for populating a form."""
    return {
        "job_families": list(skills.job_families()),
        "industries": list(skills.industries()),
        "seniority_levels": list(skills.seniority_levels()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
