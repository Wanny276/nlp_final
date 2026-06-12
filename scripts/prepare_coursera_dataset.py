"""为 CourseInsight 准备均衡的 Coursera 评价样本。

源文件应包含：

    text,label

本项目使用的标签映射：

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
    per_label: int = 3000,
    min_chars: int = 20,
    max_chars: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """读取、清洗、均衡抽样并导出 Coursera 评价。"""

    input_file = Path(input_path)
    output_file = Path(output_path)

    df = pd.read_csv(input_file)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("输入 CSV 必须包含 'text' 和 'label' 列")

    prepared = df[["text", "label"]].copy()
    prepared["text"] = prepared["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    prepared = prepared[prepared["text"].str.len() > 0]
    prepared = prepared[prepared["text"].str.len().between(min_chars, max_chars)]
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
    parser.add_argument("--per-label", type=int, default=3000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    sampled = prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        per_label=args.per_label,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        random_state=args.random_state,
    )
    print(f"已保存={args.output}")
    print(f"行数={len(sampled)}")
    print(sampled["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
