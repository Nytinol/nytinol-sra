import os
from pathlib import Path

import numpy as np
from google import genai

import tiger_db
from search_algorithm import Achievement, CurrentAchievements, search_algorithm

ENV_PATH = Path(__file__).resolve().parent / ".env"


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
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class CareerSearch:
    def recommend(
        self,
        uid: str,
        name: str,
        age: int,
        major: str,
        gpa: float,
        classes: list[str],
        salary: float,
        job_title: str,
        industry: str,
        experiences: list[dict],
    ) -> list[str]:
        job_vector, industry_vector = tiger_db.vectors_for_user(
            uid,
            job_title,
            industry,
            self._embed,
        )
        achievements = CurrentAchievements(
            name=name,
            age=age,
            major=major,
            experiences=[
                Achievement(
                    experience_type=item["experience_type"],
                    experience_name=item["experience_name"],
                    industry=item["industry"],
                    term=item["term"],
                    terms_participated=item["terms_participated"],
                    hours_per_week=item.get("hours_per_week"),
                )
                for item in experiences
            ],
        )
        steps = search_algorithm(
            job_vector,
            achievements,
            gpa,
            classes,
            major,
            salary,
            industry_vector,
        )
        return steps

    def _embed(self, text: str) -> list[float]:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
        )
        if not response.embeddings or response.embeddings[0].values is None:
            raise RuntimeError(f"Gemini returned no embedding for {text!r}")
        vector = np.array(response.embeddings[0].values, dtype=float)
        return (vector / max(float(np.linalg.norm(vector)), 1e-12)).tolist()
