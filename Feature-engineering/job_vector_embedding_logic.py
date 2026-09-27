import json
import os
import re
from pathlib import Path
from google import genai


import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "hackumbc-2026-main" / "data"
LOOKUP_PATH = DATA_DIR / "job_title_embeddings.csv"
ENV_PATH = ROOT_DIR / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])



def unique_job_titles(alumni: pd.DataFrame) -> list[str]:
    titles = alumni["first_job_title"].dropna().astype(str).unique()
    return sorted(titles)


def title_id(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def embed_texts(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
        )
        if not response.embeddings or response.embeddings[0].values is None:
            raise RuntimeError(f"Gemini returned no embedding for {text!r}")
        rows.append(response.embeddings[0].values)
    vectors = np.array(rows, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def build_job_title_lookup(alumni: pd.DataFrame) -> pd.DataFrame:
    titles = unique_job_titles(alumni)
    vectors = embed_texts(titles)
    return pd.DataFrame(
        {
            "id": [title_id(title) for title in titles],
            "text": titles,
            "maps_to": titles,
            "embedding": [json.dumps(vector.tolist()) for vector in vectors],
        }
    )


def save_job_title_lookup(lookup: pd.DataFrame, path: Path = LOOKUP_PATH) -> Path:
    lookup.to_csv(path, index=False)
    return path


def load_job_title_lookup(path: Path = LOOKUP_PATH) -> pd.DataFrame:
    lookup = pd.read_csv(path)
    lookup["embedding"] = lookup["embedding"].map(json.loads)
    return lookup


def _embedding_matrix(lookup: pd.DataFrame) -> np.ndarray:
    values = lookup["embedding"]
    if isinstance(values.iloc[0], str):
        values = values.map(json.loads)
    return np.vstack([np.asarray(value, dtype=float) for value in values])


def get_job_vector(job_title: str) -> list[float]:
    import sys

    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    import tiger_db

    return tiger_db.get_job_vector(job_title)


def nearest_job_title(goal: str, lookup: pd.DataFrame) -> str:
    goal_vector = embed_texts([goal])[0]
    scores = _embedding_matrix(lookup) @ goal_vector
    return str(lookup.iloc[int(np.argmax(scores))]["maps_to"])


def main() -> None:
    alumni = pd.read_csv(DATA_DIR / "alumni.csv")
    lookup = build_job_title_lookup(alumni)
    save_job_title_lookup(lookup)


if __name__ == "__main__":
    main()
