"""Visualization helpers used by Streamlit and reports."""

from __future__ import annotations

from pathlib import Path


def save_bar_chart(data: dict[str, int], title: str, output_path: str | Path) -> Path:
    """Save a simple bar chart."""

    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#3b82f6")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path

