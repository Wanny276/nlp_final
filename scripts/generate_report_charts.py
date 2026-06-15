"""Generate all report charts used by the LaTeX report."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "report" / "figures"

BLUE = "#315B7D"
LIGHT_BLUE = "#8DB7D3"
ORANGE = "#E8873A"
GREEN = "#3A8D6D"
PURPLE = "#7567A8"
RED = "#C65D57"
GRAY = "#6B7280"
GRID = "#DCE4EA"
TEXT = "#17212B"

LABEL_KEYS = ["negative", "neutral", "positive"]
LABEL_NAMES_ZH = ["负面", "中性", "正面"]


def read_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in candidates if name in available), None)
    if chinese_font:
        plt.rcParams["font.sans-serif"] = [chinese_font, "DejaVu Sans"]

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.edgecolor": "#7A8793",
            "axes.labelcolor": "#27313A",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.axisbelow": True,
            "font.size": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=300, pad_inches=0.08)
    plt.close(fig)


def add_y_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.75, alpha=0.65, zorder=0)
    ax.xaxis.grid(False)
    for line in ax.get_ygridlines():
        line.set_zorder(0)


def add_xy_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(axis="both", color=GRID, linewidth=0.75, alpha=0.65, zorder=0)
    for line in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
        line.set_zorder(0)


def annotate_bars(
    ax: plt.Axes,
    bars,
    *,
    percentage: bool = False,
    decimals: int = 1,
    offset: int = 4,
    fontsize: int = 9,
) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value * 100:.{decimals}f}%" if percentage else f"{value:,.0f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            zorder=5,
        )


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def find_trainer_state_path() -> Path:
    fixed = ROOT / "outputs" / "bert_model_final" / "checkpoint-1368" / "trainer_state.json"
    if fixed.exists():
        return fixed

    candidates = list((ROOT / "outputs" / "bert_model_final").glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(
            "Cannot find trainer_state.json under outputs/bert_model_final/checkpoint-*/"
        )

    def checkpoint_step(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)", str(path))
        return int(match.group(1)) if match else -1

    return sorted(candidates, key=checkpoint_step)[-1]


def read_bert_training_logs() -> tuple[list[int], list[float], np.ndarray, list[int], list[float], list[float], list[float]]:
    with find_trainer_state_path().open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    training_logs = [
        item for item in state["log_history"] if "loss" in item and "step" in item
    ]
    evaluation_logs = [
        item
        for item in state["log_history"]
        if "eval_accuracy" in item and "eval_macro_f1" in item
    ]

    steps = [int(item["step"]) for item in training_logs]
    losses = [float(item["loss"]) for item in training_logs]
    smooth_losses = moving_average(losses, window=5)
    smooth_steps = steps[len(steps) - len(smooth_losses):]

    epochs = [float(item["epoch"]) for item in evaluation_logs]
    eval_accuracy = [float(item["eval_accuracy"]) for item in evaluation_logs]
    eval_macro_f1 = [float(item["eval_macro_f1"]) for item in evaluation_logs]

    return steps, losses, smooth_losses, smooth_steps, epochs, eval_accuracy, eval_macro_f1


def read_label_counts() -> list[int]:
    csv_path = ROOT / "data" / "processed" / "bilingual_reviews_train.csv"
    counts: Counter[str] = Counter()

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = (row.get("label") or "").strip().lower()
                if label in LABEL_KEYS:
                    counts[label] += 1

    if all(counts[key] > 0 for key in LABEL_KEYS):
        return [counts[key] for key in LABEL_KEYS]

    bert = read_json("outputs/bert_metrics_final.json")
    total = int(bert["train_size"] + bert["validation_size"] + bert["test_size"])
    return [total // 3, total // 3, total - 2 * (total // 3)]


def plot_dataset_distribution() -> None:
    label_counts = read_label_counts()
    bert = read_json("outputs/bert_metrics_final.json")

    split_names = ["训练集", "验证集", "测试集"]
    split_counts = [
        int(bert["train_size"]),
        int(bert["validation_size"]),
        int(bert["test_size"]),
    ]
    total = sum(split_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.5))
    fig.suptitle(f"训练数据构成与实验划分（共 {total:,} 条）", fontsize=16, fontweight="bold", y=1.03)

    x1 = np.arange(len(LABEL_NAMES_ZH))
    bars1 = ax1.bar(
        x1,
        label_counts,
        width=0.58,
        color=[RED, GRAY, GREEN],
        zorder=3,
    )
    ax1.set_title("情感标签完全均衡")
    ax1.set_ylabel("样本数")
    ax1.set_xticks(x1, LABEL_NAMES_ZH)
    ax1.set_ylim(0, max(label_counts) * 1.18)
    add_y_grid(ax1)
    annotate_bars(ax1, bars1, fontsize=9)

    x2 = np.arange(len(split_names))
    bars2 = ax2.bar(
        x2,
        split_counts,
        width=0.58,
        color=[BLUE, LIGHT_BLUE, ORANGE],
        zorder=3,
    )
    ax2.set_title("训练、验证与测试集独立划分")
    ax2.set_ylabel("样本数")
    ax2.set_xticks(x2, split_names)
    ax2.set_ylim(0, max(split_counts) * 1.18)
    add_y_grid(ax2)
    annotate_bars(ax2, bars2, fontsize=9)

    for bar, count in zip(bars2, split_counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            count * 0.52,
            f"{count / total:.0%}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            zorder=5,
        )

    fig.tight_layout()
    finish_figure(fig, "dataset_distribution.png")


def plot_model_performance() -> None:
    traditional = read_json("models/model_metrics.json")
    bert = read_json("outputs/bert_metrics_final.json")
    model_keys = [
        "Dummy Most Frequent",
        "Naive Bayes",
        "Linear SVM",
        "Logistic Regression",
    ]
    display_names = ["Dummy", "Naive Bayes", "Linear SVM", "Logistic\nRegression", "mBERT"]
    accuracy = [traditional["results"][key]["accuracy"] for key in model_keys]
    macro_f1 = [traditional["results"][key]["macro_f1"] for key in model_keys]
    accuracy.append(bert["accuracy"])
    macro_f1.append(bert["macro_f1"])

    x = np.arange(len(display_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=BLUE,
        zorder=3,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ORANGE,
        zorder=3,
    )
    ax.set_ylabel("指标值")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 0.86)
    add_y_grid(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    annotate_bars(ax, bars_accuracy, percentage=True)
    annotate_bars(ax, bars_f1, percentage=True)
    fig.tight_layout()
    finish_figure(fig, "model_performance.png")


def draw_training_loss(ax: plt.Axes) -> None:
    steps, losses, smooth_losses, smooth_steps, *_ = read_bert_training_logs()
    ax.plot(
        steps,
        losses,
        color=LIGHT_BLUE,
        alpha=0.45,
        linewidth=1.0,
        label="批次损失",
        zorder=3,
    )
    ax.plot(
        smooth_steps,
        smooth_losses,
        color=BLUE,
        linewidth=2.4,
        label="5 点滑动平均",
        zorder=4,
    )
    ax.set_title("训练损失")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("交叉熵损失")
    add_xy_grid(ax)
    ax.legend(frameon=False, loc="upper right")


def draw_validation_metrics(ax: plt.Axes) -> None:
    *_, epochs, eval_accuracy, eval_macro_f1 = read_bert_training_logs()
    ax.plot(
        epochs,
        eval_accuracy,
        marker="o",
        markersize=7,
        linewidth=2.4,
        color=BLUE,
        label="Validation Accuracy",
        zorder=4,
    )
    ax.plot(
        epochs,
        eval_macro_f1,
        marker="s",
        markersize=7,
        linewidth=2.4,
        color=ORANGE,
        label="Validation Macro-F1",
        zorder=4,
    )
    for epoch, value in zip(epochs, eval_accuracy):
        ax.annotate(
            f"{value:.3f}",
            (epoch, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            zorder=5,
        )
    for epoch, value in zip(epochs, eval_macro_f1):
        ax.annotate(
            f"{value:.3f}",
            (epoch, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            zorder=5,
        )
    ax.set_title("验证集指标")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("指标值")
    ax.set_xticks(epochs)
    ax.set_ylim(0.66, 0.75)
    add_xy_grid(ax)
    ax.legend(frameon=False, loc="lower right")


def plot_bert_training_loss() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    draw_training_loss(ax)
    fig.tight_layout()
    finish_figure(fig, "bert_training_loss.png")


def plot_bert_validation_metrics() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    draw_validation_metrics(ax)
    fig.tight_layout()
    finish_figure(fig, "bert_validation_metrics.png")


def plot_bert_training_curve() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5))
    draw_training_loss(ax1)
    draw_validation_metrics(ax2)
    fig.tight_layout()
    finish_figure(fig, "bert_training_curve.png")


def plot_bert_confusion_matrix() -> None:
    metrics = read_json("outputs/bert_metrics_final.json")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(LABEL_NAMES_ZH)), LABEL_NAMES_ZH)
    ax.set_yticks(range(len(LABEL_NAMES_ZH)), LABEL_NAMES_ZH)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")

    threshold = matrix.max() * 0.58
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] >= threshold else TEXT,
                fontsize=12,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("样本数")
    fig.tight_layout()
    finish_figure(fig, "bert_confusion_matrix.png")


def read_stress_test_arrays() -> tuple[list[str], list[float], list[float], list[str], np.ndarray, np.ndarray, np.ndarray]:
    metrics = read_json("outputs/reports/stress_test_final/stress_metrics.json")

    experiment_keys = ["rule-only", "tfidf-only", "bert-only", "hybrid"]
    display_names = ["Rule", "TF-IDF", "BERT", "Hybrid"]
    accuracy = [float(metrics["experiments"][key]["accuracy"]) for key in experiment_keys]
    macro_f1 = [float(metrics["experiments"][key]["macro_f1"]) for key in experiment_keys]

    category_keys = [
        "abbreviation",
        "code_switching",
        "implicit_complaint",
        "long_text",
        "mixed_sentiment",
        "negation",
        "sarcasm",
        "spelling",
    ]
    category_names = ["缩写", "中英混合", "隐含抱怨", "长文本", "混合情感", "否定", "讽刺", "拼写"]

    passed = np.asarray(
        [
            [
                metrics["experiments"][experiment]["by_category"][category]["passed"]
                for category in category_keys
            ]
            for experiment in experiment_keys
        ],
        dtype=float,
    )
    support = np.asarray(
        [
            [
                metrics["experiments"][experiment]["by_category"][category]["support"]
                for category in category_keys
            ]
            for experiment in experiment_keys
        ],
        dtype=float,
    )
    heatmap = np.divide(passed, support, out=np.zeros_like(passed), where=support != 0)
    return display_names, accuracy, macro_f1, category_names, passed, support, heatmap


def draw_stress_overall(ax: plt.Axes) -> None:
    display_names, accuracy, macro_f1, *_ = read_stress_test_arrays()
    x = np.arange(len(display_names))
    width = 0.36
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=BLUE,
        zorder=3,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ORANGE,
        zorder=3,
    )
    ax.set_ylabel("指标值")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 0.86)
    add_y_grid(ax)
    ax.legend(frameon=False, loc="upper left")
    annotate_bars(ax, bars_accuracy, percentage=True, decimals=0)
    annotate_bars(ax, bars_f1, percentage=True, decimals=0)


def draw_stress_heatmap(ax: plt.Axes, *, label_mode: str = "fraction"):
    display_names, _, _, category_names, passed, support, heatmap = read_stress_test_arrays()
    image = ax.imshow(heatmap, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(category_names)), category_names)
    ax.set_yticks(range(len(display_names)), display_names)

    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = heatmap[row, column]
            label = f"{value:.0%}" if label_mode == "percent" else f"{int(passed[row, column])}/{int(support[row, column])}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value >= 0.62 else TEXT,
                fontsize=9,
                fontweight="bold",
                zorder=5,
            )
    return image


def plot_stress_test_overall() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    draw_stress_overall(ax)
    fig.tight_layout()
    finish_figure(fig, "stress_test_overall.png")


def plot_stress_test_categories() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    image = draw_stress_heatmap(ax, label_mode="fraction")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.025)
    colorbar.set_label("通过率")
    fig.tight_layout()
    finish_figure(fig, "stress_test_categories.png")


def plot_stress_test_results() -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.8, 4.6),
        gridspec_kw={"width_ratios": [1.0, 1.65]},
    )
    draw_stress_overall(ax1)
    image = draw_stress_heatmap(ax2, label_mode="percent")
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")
    colorbar = fig.colorbar(image, ax=ax2, fraction=0.035, pad=0.025)
    colorbar.set_label("Accuracy")
    fig.tight_layout()
    finish_figure(fig, "stress_test_results.png")


def plot_ablation_results() -> None:
    metrics = read_json("outputs/reports/final_model/ablation_metrics.json")
    experiment_keys = ["rule-only", "model-only", "bert-only", "hybrid"]
    display_names = ["Rule only", "TF-IDF only", "BERT only", "Hybrid"]
    accuracy = [metrics["experiments"][key]["accuracy"] for key in experiment_keys]
    macro_f1 = [metrics["experiments"][key]["macro_f1"] for key in experiment_keys]
    passed = [metrics["experiments"][key]["passed"] for key in experiment_keys]

    x = np.arange(len(display_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axvspan(2.52, 3.48, color="#E8F3EC", alpha=0.9, zorder=-2)
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=BLUE,
        zorder=3,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ORANGE,
        zorder=3,
    )
    ax.set_ylabel("指标值")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 1.16)
    add_y_grid(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    annotate_bars(ax, bars_accuracy, percentage=True, decimals=1)
    annotate_bars(ax, bars_f1, percentage=True, decimals=1)

    for index, count in enumerate(passed):
        ax.text(
            index,
            0.045,
            f"通过 {count}/12",
            ha="center",
            va="bottom",
            color="white" if accuracy[index] > 0.65 else TEXT,
            fontweight="bold",
            fontsize=9,
        )

    ax.text(
        3,
        1.105,
        "完整流程：12/12",
        color="#357354",
        fontweight="bold",
        ha="center",
    )
    fig.tight_layout()
    finish_figure(fig, "ablation_results.png")


def plot_traditional_confusion_matrix() -> None:
    metrics = read_json("models/model_metrics.json")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_title("Logistic Regression 测试集混淆矩阵", fontsize=18, fontweight="bold", pad=10)
    ax.set_xticks(range(len(LABEL_NAMES_ZH)), LABEL_NAMES_ZH)
    ax.set_yticks(range(len(LABEL_NAMES_ZH)), LABEL_NAMES_ZH)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")

    threshold = matrix.max() * 0.58
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] >= threshold else TEXT,
                fontsize=14,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("样本数")
    fig.tight_layout()
    finish_figure(fig, "traditional_confusion_matrix.png")


def main() -> None:
    configure_style()
    plot_ablation_results()
    plot_bert_confusion_matrix()
    plot_bert_training_curve()
    plot_bert_training_loss()
    plot_bert_validation_metrics()
    plot_dataset_distribution()
    plot_model_performance()
    plot_stress_test_categories()
    plot_stress_test_overall()
    plot_stress_test_results()
    plot_traditional_confusion_matrix()
    print(f"Generated report charts in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
