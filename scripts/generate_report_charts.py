"""Generate reproducible charts used by the LaTeX report."""

from __future__ import annotations

import json
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
GRID = "#D9E1E8"


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


def annotate_bars(
    ax: plt.Axes,
    bars,
    *,
    percentage: bool = False,
    decimals: int = 1,
) -> None:
    for bar in bars:
        value = bar.get_height()
        label = (
            f"{value * 100:.{decimals}f}%"
            if percentage
            else f"{value:,.0f}"
        )
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


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
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ORANGE,
    )
    ax.set_ylabel("指标值")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 0.86)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    annotate_bars(ax, bars_accuracy, percentage=True)
    annotate_bars(ax, bars_f1, percentage=True)
    fig.tight_layout()
    finish_figure(fig, "model_performance.png")


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_bert_training_figures() -> None:
    state = read_json(
        "outputs/bert_model_final/checkpoint-1368/trainer_state.json"
    )
    training_logs = [
        item for item in state["log_history"] if "loss" in item and "step" in item
    ]
    evaluation_logs = [
        item
        for item in state["log_history"]
        if "eval_accuracy" in item and "eval_macro_f1" in item
    ]
    steps = [item["step"] for item in training_logs]
    losses = [item["loss"] for item in training_logs]
    smooth_losses = moving_average(losses)
    smooth_steps = steps[len(steps) - len(smooth_losses) :]

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot(
        steps,
        losses,
        color=LIGHT_BLUE,
        alpha=0.5,
        linewidth=1.0,
        label="批次损失",
    )
    ax.plot(
        smooth_steps,
        smooth_losses,
        color=BLUE,
        linewidth=2.2,
        label="5 点滑动平均",
    )
    ax.set_xlabel("训练步数")
    ax.set_ylabel("交叉熵损失")
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    finish_figure(fig, "bert_training_loss.png")

    epochs = [item["epoch"] for item in evaluation_logs]
    eval_accuracy = [item["eval_accuracy"] for item in evaluation_logs]
    eval_macro_f1 = [item["eval_macro_f1"] for item in evaluation_logs]
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot(
        epochs,
        eval_accuracy,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=BLUE,
        label="Validation Accuracy",
    )
    ax.plot(
        epochs,
        eval_macro_f1,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color=ORANGE,
        label="Validation Macro-F1",
    )
    for epoch, value in zip(epochs, eval_accuracy):
        ax.annotate(
            f"{value:.3f}",
            (epoch, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    for epoch, value in zip(epochs, eval_macro_f1):
        ax.annotate(
            f"{value:.3f}",
            (epoch, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("指标值")
    ax.set_xticks(epochs)
    ax.set_ylim(0.66, 0.75)
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    finish_figure(fig, "bert_validation_metrics.png")


def plot_bert_confusion_matrix() -> None:
    metrics = read_json("outputs/bert_metrics_final.json")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    labels = ["负面", "中性", "正面"]

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
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
                color="white" if matrix[row, column] >= threshold else "#17212B",
                fontsize=12,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("样本数")
    fig.tight_layout()
    finish_figure(fig, "bert_confusion_matrix.png")


def plot_stress_test_results() -> None:
    metrics = read_json(
        "outputs/reports/stress_test_final/stress_metrics.json"
    )
    experiment_keys = ["rule-only", "tfidf-only", "bert-only", "hybrid"]
    display_names = ["Rule", "TF-IDF", "BERT", "Hybrid"]
    accuracy = [metrics["experiments"][key]["accuracy"] for key in experiment_keys]
    macro_f1 = [metrics["experiments"][key]["macro_f1"] for key in experiment_keys]
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
        ]
    )
    support = np.asarray(
        [
            [
                metrics["experiments"][experiment]["by_category"][category]["support"]
                for category in category_keys
            ]
            for experiment in experiment_keys
        ]
    )
    heatmap = passed / support

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(display_names))
    width = 0.36
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=BLUE,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ORANGE,
    )
    ax.set_ylabel("指标值")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 0.86)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    annotate_bars(ax, bars_accuracy, percentage=True, decimals=0)
    annotate_bars(ax, bars_f1, percentage=True, decimals=0)
    fig.tight_layout()
    finish_figure(fig, "stress_test_overall.png")

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    image = ax.imshow(heatmap, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(category_names)), category_names)
    ax.set_yticks(range(len(display_names)), display_names)
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = heatmap[row, column]
            ax.text(
                column,
                row,
                f"{passed[row, column]}/{support[row, column]}",
                ha="center",
                va="center",
                color="white" if value >= 0.62 else "#17212B",
                fontsize=9,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.025)
    colorbar.set_label("通过率")
    fig.tight_layout()
    finish_figure(fig, "stress_test_categories.png")


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
    ax.axvspan(2.52, 3.48, color="#E8F3EC", alpha=0.9, zorder=0)
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=BLUE,
        zorder=2,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ORANGE,
        zorder=2,
    )
    ax.set_ylabel("指标值")
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 1.16)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=1)
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
            color="white" if accuracy[index] > 0.65 else "#17212B",
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


def main() -> None:
    configure_style()
    plot_model_performance()
    plot_bert_training_figures()
    plot_bert_confusion_matrix()
    plot_ablation_results()
    plot_stress_test_results()
    print(f"Generated report charts in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
