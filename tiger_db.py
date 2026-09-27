import asyncio
import json
import os
from pathlib import Path

import asyncpg

ROOT_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = ROOT_DIR / "tiger-cloud-nytinoltiger-credentials.env"


def load_env(path: Path = CREDENTIALS_PATH) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


async def _fetch_embedding(table: str, label: str) -> list[float]:
    load_env()
    conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
    try:
        row = await conn.fetchrow(
            f"""
            SELECT embedding
            FROM {table}
            WHERE maps_to = $1 OR text = $1 OR id = $1
            """,
            label,
        )
    finally:
        await conn.close()
    if row is None:
        raise KeyError(f"No vector stored in {table} for {label!r}")
    return [float(value) for value in json.loads(row["embedding"])]


def get_job_vector(job_title: str) -> list[float]:
    return asyncio.run(_fetch_embedding("job_title_embeddings", job_title))


def get_industry_vector(industry: str) -> list[float]:
    return asyncio.run(_fetch_embedding("industry_embeddings", industry))


async def _fetch_all_job_vectors() -> list[list[float]]:
    load_env()
    conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
    try:
        rows = await conn.fetch(
            "SELECT embedding FROM job_title_embeddings ORDER BY maps_to"
        )
    finally:
        await conn.close()
    return [[float(value) for value in json.loads(row["embedding"])] for row in rows]


def get_all_job_vectors() -> list[list[float]]:
    return asyncio.run(_fetch_all_job_vectors())


def _vector(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in json.loads(value)]


async def _load_scoring_data() -> tuple[list[dict], dict[str, list[float]], dict[str, list[float]]]:
    load_env()
    conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
    try:
        alumni_rows = await conn.fetch(
            """
            SELECT
                campus_id,
                major,
                track,
                final_gpa,
                major_gpa,
                first_job_title,
                first_employer_industry,
                first_job_annual_salary_usd,
                first_job_found_via,
                months_to_first_job,
                internship_count
            FROM alumni
            """
        )
        title_rows = await conn.fetch("SELECT maps_to, embedding FROM job_title_embeddings")
        industry_rows = await conn.fetch("SELECT maps_to, embedding FROM industry_embeddings")
    finally:
        await conn.close()
    titles = {row["maps_to"]: _vector(row["embedding"]) for row in title_rows}
    industries = {row["maps_to"]: _vector(row["embedding"]) for row in industry_rows}
    alumni = [
        {
            "campus_id": row["campus_id"],
            "major": row["major"],
            "track": row["track"],
            "final_gpa": float(row["final_gpa"]),
            "major_gpa": float(row["major_gpa"]),
            "first_job_title": row["first_job_title"],
            "first_employer_industry": row["first_employer_industry"],
            "first_job_annual_salary_usd": float(row["first_job_annual_salary_usd"]),
            "first_job_found_via": row["first_job_found_via"],
            "months_to_first_job": float(row["months_to_first_job"]),
            "internship_count": int(row["internship_count"]),
        }
        for row in alumni_rows
        if row["first_job_title"] in titles and row["first_employer_industry"] in industries
    ]
    return alumni, titles, industries


_scoring_cache: tuple[list[dict], dict[str, list[float]], dict[str, list[float]]] | None = None


def load_scoring_data() -> tuple[list[dict], dict[str, list[float]], dict[str, list[float]]]:
    global _scoring_cache
    if _scoring_cache is None:
        _scoring_cache = asyncio.run(_load_scoring_data())
    return _scoring_cache


async def _load_campus_records(campus_ids: list[str]) -> dict[str, list[dict]]:
    load_env()
    conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
    try:
        alumni = await conn.fetch(
            "SELECT * FROM alumni WHERE campus_id = ANY($1::text[])",
            campus_ids,
        )
        employment = await conn.fetch(
            "SELECT * FROM employment_history WHERE campus_id = ANY($1::text[])",
            campus_ids,
        )
        experience = await conn.fetch(
            "SELECT * FROM student_experience WHERE campus_id = ANY($1::text[])",
            campus_ids,
        )
        transcripts = await conn.fetch(
            "SELECT * FROM transcripts WHERE campus_id = ANY($1::text[])",
            campus_ids,
        )
        course_ids = list({row["course_id"] for row in transcripts})
        catalog = await conn.fetch(
            "SELECT * FROM course_catalog WHERE course_id = ANY($1::text[])",
            course_ids,
        )
    finally:
        await conn.close()
    return {
        "alumni": [dict(row) for row in alumni],
        "employment_history": [dict(row) for row in employment],
        "student_experience": [dict(row) for row in experience],
        "transcripts": [dict(row) for row in transcripts],
        "course_catalog": [dict(row) for row in catalog],
    }


def load_campus_records(campus_ids: list[str]) -> dict[str, list[dict]]:
    return asyncio.run(_load_campus_records(campus_ids))


async def _ensure_goal_tables(conn: asyncpg.Connection) -> None:
    await conn.execute("DROP TABLE IF EXISTS user_goal_vectors")
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_vectors (
            position SERIAL PRIMARY KEY,
            job_title TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS industry_vectors (
            position SERIAL PRIMARY KEY,
            industry TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            job_position INTEGER NOT NULL REFERENCES job_vectors(position),
            industry_position INTEGER NOT NULL REFERENCES industry_vectors(position)
        )
        """
    )


async def _catalog_vector(conn: asyncpg.Connection, table: str, label: str) -> list[float] | None:
    row = await conn.fetchrow(
        f"""
        SELECT embedding
        FROM {table}
        WHERE maps_to = $1 OR text = $1 OR id = $1
        """,
        label,
    )
    if row is None:
        return None
    return _vector(row["embedding"])


async def _vectors_for_user(
    uid: str,
    job_title: str,
    industry: str,
    embed,
) -> tuple[list[float], list[float]]:
    load_env()
    conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
    try:
        await _ensure_goal_tables(conn)
        job_row = await conn.fetchrow(
            "SELECT position, embedding FROM job_vectors WHERE job_title = $1",
            job_title,
        )
        if job_row is not None:
            job_vector = _vector(job_row["embedding"])
            job_position = job_row["position"]
        else:
            job_vector = await _catalog_vector(conn, "job_title_embeddings", job_title)
            if job_vector is None:
                job_vector = embed(job_title)
            job_position = await conn.fetchval(
                """
                INSERT INTO job_vectors (job_title, embedding)
                VALUES ($1, $2)
                ON CONFLICT (job_title) DO UPDATE SET job_title = EXCLUDED.job_title
                RETURNING position
                """,
                job_title,
                json.dumps(job_vector),
            )
        industry_row = await conn.fetchrow(
            "SELECT position, embedding FROM industry_vectors WHERE industry = $1",
            industry,
        )
        if industry_row is not None:
            industry_vector = _vector(industry_row["embedding"])
            industry_position = industry_row["position"]
        else:
            industry_vector = await _catalog_vector(conn, "industry_embeddings", industry)
            if industry_vector is None:
                industry_vector = embed(industry)
            industry_position = await conn.fetchval(
                """
                INSERT INTO industry_vectors (industry, embedding)
                VALUES ($1, $2)
                ON CONFLICT (industry) DO UPDATE SET industry = EXCLUDED.industry
                RETURNING position
                """,
                industry,
                json.dumps(industry_vector),
            )
        await conn.execute(
            """
            INSERT INTO users (uid, job_position, industry_position)
            VALUES ($1, $2, $3)
            ON CONFLICT (uid) DO UPDATE SET
                job_position = EXCLUDED.job_position,
                industry_position = EXCLUDED.industry_position
            """,
            uid,
            job_position,
            industry_position,
        )
    finally:
        await conn.close()
    if job_vector is None or industry_vector is None:
        raise RuntimeError("Goal vectors were not stored")
    return job_vector, industry_vector


def vectors_for_user(
    uid: str,
    job_title: str,
    industry: str,
    embed,
) -> tuple[list[float], list[float]]:
    return asyncio.run(_vectors_for_user(uid, job_title, industry, embed))
