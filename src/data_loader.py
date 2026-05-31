"""Data loading and validation."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_TEXT_COLUMN = "text"


def load_reviews_csv(path: str | Path) -> list[dict[str, str]]:
    """Load review rows from a CSV file with a required text column."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or REQUIRED_TEXT_COLUMN not in reader.fieldnames:
            raise ValueError(f"CSV must include a '{REQUIRED_TEXT_COLUMN}' column")

        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in reader:
            text = (row.get(REQUIRED_TEXT_COLUMN) or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            rows.append({key: (value or "").strip() for key, value in row.items()})

    return rows


def rows_to_texts(rows: list[dict[str, str]]) -> list[str]:
    """Extract review texts from loaded rows."""

    return [row[REQUIRED_TEXT_COLUMN] for row in rows if row.get(REQUIRED_TEXT_COLUMN)]

