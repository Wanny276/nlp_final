"""Prepare a balanced Coursera review sample for CourseInsight.

The source file is expected to contain:

    text,label

Label mapping used by this project:

    0 -> negative
    1 -> neutral
    2 -> positive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    per_label: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load, clean, balance, and export Coursera reviews."""

    input_file = Path(input_path)
    output_file = Path(output_path)

    df = pd.read_csv(input_file)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")

    prepared = df[["text", "label"]].copy()
    prepared["text"] = prepared["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    prepared = prepared[prepared["text"].str.len() > 0]
    prepared = prepared.drop_duplicates(subset=["text"])
    prepared["source_label"] = prepared["label"]
    prepared["label"] = prepared["label"].map(LABEL_MAP)
    prepared = prepared.dropna(subset=["label"])

    samples = []
    for label in ["positive", "neutral", "negative"]:
        label_df = prepared[prepared["label"] == label]
        sample_size = min(per_label, len(label_df))
        samples.append(label_df.sample(n=sample_size, random_state=random_state))

    sampled = pd.concat(samples, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sampled.insert(0, "id", range(1, len(sampled) + 1))
    sampled["course"] = "Coursera Online Course"
    sampled["teacher"] = "Coursera Instructor"
    sampled = sampled[["id", "text", "course", "teacher", "label", "source_label"]]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_file, index=False, encoding="utf-8-sig")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/coursera_reviews_label_3.csv")
    parser.add_argument("--output", default="data/processed/coursera_reviews_sampled.csv")
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    sampled = prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        per_label=args.per_label,
        random_state=args.random_state,
    )
    print(f"saved={args.output}")
    print(f"rows={len(sampled)}")
    print(sampled["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
