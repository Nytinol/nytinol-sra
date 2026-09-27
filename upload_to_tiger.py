import asyncio
import os
import re
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "hackumbc-2026-main" / "data"
CREDENTIALS_PATH = ROOT_DIR / "tiger-cloud-nytinoltiger-credentials.env"


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def table_name(csv_path: Path) -> str:
    name = re.sub(r"[^a-z0-9_]", "_", csv_path.stem.lower()).strip("_")
    if not name or name[0].isdigit():
        name = f"csv_{name}"
    return name


def column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    return "TEXT"


def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def records_for(df: pd.DataFrame) -> list[tuple]:
    rows = []
    for record in df.itertuples(index=False, name=None):
        values = []
        for value in record:
            if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                values.append(None)
            elif isinstance(value, np.generic):
                values.append(value.item())
            else:
                values.append(value)
        rows.append(tuple(values))
    return rows


async def upload_csv(conn: asyncpg.Connection, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    name = table_name(csv_path)
    columns = list(df.columns)
    column_sql = ", ".join(
        f"{sql_identifier(column)} {column_type(df[column])}" for column in columns
    )
    await conn.execute(f"DROP TABLE IF EXISTS {sql_identifier(name)}")
    await conn.execute(f"CREATE TABLE {sql_identifier(name)} ({column_sql})")
    if len(df) == 0:
        print(f"{name}: 0 rows")
        return
    await conn.copy_records_to_table(name, records=records_for(df), columns=columns)
    print(f"{name}: {len(df)} rows")


async def main() -> None:
    load_env(CREDENTIALS_PATH)
    conn = await asyncpg.connect(os.environ["TIMESCALE_SERVICE_URL"])
    try:
        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            await upload_csv(conn, csv_path)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
