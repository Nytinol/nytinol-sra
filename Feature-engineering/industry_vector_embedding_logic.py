import asyncio
import json
import os
import re
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd
from google import genai

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "hackumbc-2026-main" / "data"
ENV_PATH = ROOT_DIR / ".env"
CREDENTIALS_PATH = ROOT_DIR / "tiger-cloud-nytinoltiger-credentials.env"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class IndustryEmbeddings:
    TABLE = "industry_embeddings"
    MODEL = "gemini-embedding-2"

    def __init__(self) -> None:
        load_env(ENV_PATH)
        load_env(CREDENTIALS_PATH)
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def unique_industries(self) -> list[str]:
        names = set()
        for filename, column in (
            ("alumni.csv", "first_employer_industry"),
            ("employment_history.csv", "employer_industry"),
        ):
            frame = pd.read_csv(DATA_DIR / filename)
            values = frame[column].dropna().astype(str)
            names.update(value for value in values if value != "Not Applicable")
        return sorted(names)

    @staticmethod
    def industry_id(industry: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", industry.lower()).strip("-")

    def embed(self, industries: list[str]) -> np.ndarray:
        rows = []
        for industry in industries:
            response = self.client.models.embed_content(
                model=self.MODEL,
                contents=industry,
            )
            if not response.embeddings or response.embeddings[0].values is None:
                raise RuntimeError(f"Gemini returned no embedding for {industry!r}")
            rows.append(response.embeddings[0].values)
        vectors = np.array(rows, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def lookup_frame(self) -> pd.DataFrame:
        industries = self.unique_industries()
        vectors = self.embed(industries)
        return pd.DataFrame(
            {
                "id": [self.industry_id(industry) for industry in industries],
                "text": industries,
                "maps_to": industries,
                "embedding": [json.dumps(vector.tolist()) for vector in vectors],
            }
        )

    async def deploy(self) -> int:
        lookup = self.lookup_frame()
        conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
            await conn.execute(
                f"""
                CREATE TABLE {self.TABLE} (
                    id TEXT,
                    text TEXT,
                    maps_to TEXT,
                    embedding TEXT
                )
                """
            )
            await conn.copy_records_to_table(
                self.TABLE,
                records=list(lookup.itertuples(index=False, name=None)),
                columns=["id", "text", "maps_to", "embedding"],
            )
        finally:
            await conn.close()
        return len(lookup)


if __name__ == "__main__":
    count = asyncio.run(IndustryEmbeddings().deploy())
    print(f"industry_embeddings: {count} rows")
