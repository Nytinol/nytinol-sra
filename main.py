from fastapi import FastAPI
from pydantic import BaseModel

from api import CareerSearch

app = FastAPI()
career_search = CareerSearch()


class ExperienceIn(BaseModel):
    experience_type: str
    experience_name: str
    industry: str
    term: str
    terms_participated: int
    hours_per_week: float | None = None


class SearchRequest(BaseModel):
    uid: str
    name: str
    age: int
    major: str
    gpa: float
    classes: list[str]
    salary: float
    job_title: str
    industry: str
    experiences: list[ExperienceIn] = []


@app.post("/search")
def search(body: SearchRequest) -> list[str]:
    return career_search.recommend(
        uid=body.uid,
        name=body.name,
        age=body.age,
        major=body.major,
        gpa=body.gpa,
        classes=body.classes,
        salary=body.salary,
        job_title=body.job_title,
        industry=body.industry,
        experiences=[item.model_dump() for item in body.experiences],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
