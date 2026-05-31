"""Data loading and validation."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_TEXT_COLUMN = "text"
TEXT_COLUMN_CANDIDATES = ("text", "review", "reviews", "comment", "content", "评价", "评论")
COURSE_COLUMN_CANDIDATES = ("course", "course_title", "course_name", "课程")
TEACHER_COLUMN_CANDIDATES = ("teacher", "instructor", "讲师", "教师")
RATING_COLUMN_CANDIDATES = ("rating", "stars", "score", "评分")


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def label_from_rating(value: str) -> str:
    """Convert 1-5 star ratings to sentiment labels."""

    try:
        rating = float(value)
    except (TypeError, ValueError):
        return ""

    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def load_reviews_csv(path: str | Path) -> list[dict[str, str]]:
    """Load review rows from a CSV file with a required text column."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")

        text_column = _find_column(reader.fieldnames, TEXT_COLUMN_CANDIDATES)
        if text_column is None:
            raise ValueError("CSV must include one of these text columns: text, review, reviews, comment, content")

        course_column = _find_column(reader.fieldnames, COURSE_COLUMN_CANDIDATES)
        teacher_column = _find_column(reader.fieldnames, TEACHER_COLUMN_CANDIDATES)
        rating_column = _find_column(reader.fieldnames, RATING_COLUMN_CANDIDATES)

        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in reader:
            text = (row.get(text_column) or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            label = (row.get("label") or "").strip()
            if not label and rating_column is not None:
                label = label_from_rating(row.get(rating_column, ""))

            normalized = {key: (value or "").strip() for key, value in row.items()}
            normalized[REQUIRED_TEXT_COLUMN] = text
            normalized["course"] = (row.get(course_column) or "").strip() if course_column else ""
            normalized["teacher"] = (row.get(teacher_column) or "").strip() if teacher_column else ""
            normalized["label"] = label
            rows.append(normalized)

    return rows


def rows_to_texts(rows: list[dict[str, str]]) -> list[str]:
    """Extract review texts from loaded rows."""

    return [row[REQUIRED_TEXT_COLUMN] for row in rows if row.get(REQUIRED_TEXT_COLUMN)]
