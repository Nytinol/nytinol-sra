"""Shared paths and the one rule these scripts follow.

`data/raw/` is the pristine dataset and is never written to -- it is checked out
read-only on purpose. Every script here reads from it and writes a derived copy to
`data/derived/`, so a cleanup step can be re-run or corrected without destroying
the source.

The earlier versions of these scripts overwrote the CSVs in place, and one of them
filtered `role_level != "Not Applicable"`, which silently deleted every
internship, co-op and certification row, because those experience types have no
role level. That is 7,629 of 20,059 rows, and they are precisely the
opportunities the recommender exists to suggest.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DIR = ROOT_DIR / "data" / "raw"
DERIVED_DIR = ROOT_DIR / "data" / "derived"


def write(frame, name: str) -> Path:
    """Write a derived table, refusing to touch the raw directory."""
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = (DERIVED_DIR / f"{name}.csv").resolve()
    if RAW_DIR.resolve() in path.parents:
        raise RuntimeError(f"Refusing to write into the raw dataset: {path}")
    frame.to_csv(path, index=False)
    print(f"{name}: {len(frame)} rows, {len(frame.columns)} columns -> {path.parent.name}/{path.name}")
    return path
