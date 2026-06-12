"""构建清洗后的 Coursera 数据和双语训练集。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_ORDER = ["positive", "neutral", "negative"]


def clean_coursera(
    input_path: str | Path,
    output_path: str | Path,
    per_label: int = 3000,
    min_chars: int = 20,
    max_chars: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """筛选 Coursera 样本并保留均衡子集。"""

    df = pd.read_csv(input_path)
    required = {"text", "course", "teacher", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coursera 样本缺少列：{sorted(missing)}")

    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    cleaned = cleaned[cleaned["text"].str.len().between(min_chars, max_chars)]
    cleaned = cleaned.drop_duplicates(subset=["text"])
    cleaned = cleaned[cleaned["label"].isin(LABEL_ORDER)]
    cleaned["topics"] = ""
    cleaned["language"] = "en"
    cleaned["source"] = "coursera"

    balanced_parts = []
    for label in LABEL_ORDER:
        label_df = cleaned[cleaned["label"] == label]
        sample_size = min(per_label, len(label_df))
        balanced_parts.append(label_df.sample(n=sample_size, random_state=random_state))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    balanced.insert(0, "new_id", range(1, len(balanced) + 1))
    balanced["id"] = balanced["new_id"]
    balanced = balanced.drop(columns=["new_id"])

    columns = ["id", "text", "course", "teacher", "label", "topics", "language", "source"]
    if "source_label" in balanced.columns:
        columns.append("source_label")
    balanced = balanced[columns]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_file, index=False, encoding="utf-8-sig")
    return balanced


def build_bilingual(
    chinese_path: str | Path,
    coursera_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """合并中文人工数据和清洗后的 Coursera 数据。"""

    chinese = pd.read_csv(chinese_path)
    coursera = pd.read_csv(coursera_path)

    columns = ["text", "course", "teacher", "label", "topics", "language", "source"]
    for frame_name, frame in {"chinese": chinese, "coursera": coursera}.items():
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} 数据缺少列：{sorted(missing)}")

    merged = pd.concat([chinese[columns], coursera[columns]], ignore_index=True)
    merged["text"] = merged["text"].fillna("").astype(str).str.strip()
    merged = merged[merged["text"].str.len() > 0]
    merged = merged.drop_duplicates(subset=["text"])
    merged = merged[merged["label"].isin(LABEL_ORDER)]
    merged.insert(0, "id", range(1, len(merged) + 1))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False, encoding="utf-8-sig")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chinese", default="data/processed/chinese_manual_reviews.csv")
    parser.add_argument("--coursera", default="data/processed/coursera_reviews_sampled.csv")
    parser.add_argument("--cleaned-coursera", default="data/processed/coursera_reviews_cleaned.csv")
    parser.add_argument("--output", default="data/processed/bilingual_reviews_train.csv")
    parser.add_argument("--per-label", type=int, default=3000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    cleaned = clean_coursera(
        input_path=args.coursera,
        output_path=args.cleaned_coursera,
        per_label=args.per_label,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    merged = build_bilingual(
        chinese_path=args.chinese,
        coursera_path=args.cleaned_coursera,
        output_path=args.output,
    )

    print(f"清洗后 Coursera 数据={args.cleaned_coursera}")
    print(cleaned["label"].value_counts().to_string())
    print(f"双语训练集={args.output}")
    print(merged.groupby(["language", "label"]).size().to_string())


if __name__ == "__main__":
    main()
