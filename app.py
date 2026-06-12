"""CourseInsight 的 Streamlit 入口。"""

from __future__ import annotations

import hashlib
import html
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_reviews_csv, rows_to_texts
from src.keyword_extractor import extract_keywords
from src.llm_client import LLMConfig
from src.nlp_analyzer import analyze_batch, analyze_review, sentiment_distribution, topic_distribution
from src.topic_analyzer import detect_topics


SAMPLE_DATA = Path("data/sample_reviews.csv")
TEST_CASES = Path("data/test_cases.csv")
MODEL_METRICS = Path("models/model_metrics.json")
BERT_METRICS = Path("outputs/bert_metrics.json")
ABLATION_METRICS = Path("outputs/reports/ablation_metrics.json")

SENTIMENT_LABELS = {
    "positive": "正面 positive",
    "neutral": "中性 neutral",
    "negative": "负面 negative",
}
LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英文",
    "mixed": "中英混合",
    "unknown": "未知",
}
SENTIMENT_SOURCE_LABELS = {
    "model": "模型预测",
    "rule": "规则兜底",
    "hybrid": "规则校正",
}
SENTIMENT_UI = {
    "positive": {
        "label": "正面反馈",
        "description": "评价整体认可课程体验，可继续保持并推广优势。",
        "tone": "positive",
    },
    "neutral": {
        "label": "中性 / 混合反馈",
        "description": "评价同时包含认可与改进意见，建议分项处理。",
        "tone": "neutral",
    },
    "negative": {
        "label": "负面反馈",
        "description": "评价指出了明确问题，建议优先安排改进。",
        "tone": "negative",
    },
}
RISK_LABELS = {
    "low": ("低风险", "positive"),
    "middle": ("中风险", "neutral"),
    "high": ("高风险", "negative"),
}
PAGE_OPTIONS = [
    "评价分析",
    "批量分析",
    "测试验证",
    "系统信息",
]
SAMPLE_REVIEWS = {
    "中文评价": "老师讲得很清楚，但是作业有点多，实验环境配置也比较麻烦。",
    "English review": (
        "The instructor explains concepts clearly but the assignments are too many "
        "and the setup is confusing."
    ),
    "中英混合": "老师讲解很 clear，但是 assignment 太多，deadline 有点紧。",
}
ICON_SVGS = {
    "analysis": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 19V9m5 10V5m5 14v-7m5 7V3"/>
            <path d="m3 7 5-4 5 5 7-6"/>
        </svg>
    """,
    "batch": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="4" width="18" height="5" rx="1.5"/>
            <rect x="3" y="15" width="18" height="5" rx="1.5"/>
            <path d="M7 9v6m10-6v6"/>
        </svg>
    """,
    "check": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="9"/>
            <path d="m8 12 2.5 2.5L16.5 8.5"/>
        </svg>
    """,
    "system": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="5" y="5" width="14" height="14" rx="2"/>
            <path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m14 0h4M1 15h4m14 0h4"/>
            <rect x="9" y="9" width="6" height="6" rx="1"/>
        </svg>
    """,
    "text": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 6h14M8 10h8M6 14h12M9 18h6"/>
        </svg>
    """,
    "settings": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 7h10m4 0h2M4 17h3m4 0h9M14 4v6M7 14v6"/>
            <circle cx="16" cy="7" r="2"/>
            <circle cx="9" cy="17" r="2"/>
        </svg>
    """,
    "upload": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 16V4m0 0L8 8m4-4 4 4"/>
            <path d="M5 14v5h14v-5"/>
        </svg>
    """,
    "data": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <ellipse cx="12" cy="5" rx="8" ry="3"/>
            <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>
            <path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/>
        </svg>
    """,
    "language": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="9"/>
            <path d="M3 12h18M12 3c2.5 2.4 4 5.5 4 9s-1.5 6.6-4 9c-2.5-2.4-4-5.5-4-9s1.5-6.6 4-9Z"/>
        </svg>
    """,
    "topic": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="7" cy="7" r="3"/>
            <circle cx="17" cy="7" r="3"/>
            <circle cx="12" cy="17" r="3"/>
            <path d="m9.5 8.5 2 5m3-5-2 5"/>
        </svg>
    """,
    "keyword": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="8" cy="12" r="4"/>
            <path d="M12 12h9m-3 0v3m-3-3v2"/>
        </svg>
    """,
    "model": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3 4 7v10l8 4 8-4V7l-8-4Z"/>
            <path d="m4 7 8 4 8-4m-8 4v10"/>
        </svg>
    """,
    "chart": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 20V10m6 10V4m6 16v-7m4 7H2"/>
        </svg>
    """,
    "spark": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m12 3 1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3Z"/>
            <path d="m18.5 15 .8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z"/>
        </svg>
    """,
}

APP_STYLES = """
<style>
:root {
    --ink: #172033;
    --muted: #667085;
    --line: #e5e9f2;
    --surface: #ffffff;
    --canvas: #f5f7fb;
    --primary: #4f46e5;
    --primary-soft: #eef2ff;
    --positive: #16855b;
    --positive-soft: #eaf8f1;
    --neutral: #b56b08;
    --neutral-soft: #fff6df;
    --negative: #c43d4b;
    --negative-soft: #fff0f1;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #f6f7fb;
    color: var(--ink);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1240px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
    background:
        radial-gradient(circle at 20% 0%, rgba(129, 140, 248, 0.18), transparent 14rem),
        linear-gradient(180deg, #151d35 0%, #10162a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.45rem;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    min-height: 2.85rem;
    padding: 0.62rem 0.8rem;
    border: 1px solid transparent;
    border-radius: 0.85rem;
    background: rgba(255,255,255,0.035);
    transition: all 160ms ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.09);
    border-color: rgba(255,255,255,0.12);
    transform: translateX(2px);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(99,102,241,0.95), rgba(79,70,229,0.9));
    border-color: rgba(255,255,255,0.28);
    box-shadow: 0 10px 30px rgba(49,46,129,0.32);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-weight: 650;
    font-size: 0.96rem;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9);
    border: 1px solid var(--line);
    border-radius: 1rem;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 26px rgba(23,32,51,0.05);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--ink);
    font-weight: 750;
}

[data-testid="stForm"], [data-testid="stFileUploader"] {
    border-radius: 1rem;
}

[data-testid="stTextArea"] textarea,
[data-testid="stFileUploaderDropzone"] {
    border-radius: 0.85rem;
    border-color: #d7ddea;
    background: rgba(255,255,255,0.9);
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stDownloadButton > button {
    min-height: 2.9rem;
    border-radius: 0.75rem;
    font-weight: 650;
    transition: transform 150ms ease, box-shadow 150ms ease;
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    border: 0;
    background: linear-gradient(135deg, #5b55ea, #4338ca);
    box-shadow: 0 9px 22px rgba(79,70,229,0.22);
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem;
    padding: 0.35rem;
    border-radius: 0.85rem;
    background: #eef1f7;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 2.6rem;
    padding: 0 1rem;
    border-radius: 0.65rem;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: white;
    box-shadow: 0 3px 12px rgba(23,32,51,0.08);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 0.85rem;
    overflow: hidden;
}

.brand-card {
    padding: 0.35rem 0.4rem 1.1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.1rem;
}

.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.35rem;
    height: 2.35rem;
    margin-bottom: 0.7rem;
    border-radius: 0.8rem;
    background: linear-gradient(135deg, #818cf8, #4f46e5);
    color: white;
    font-weight: 800;
    letter-spacing: -0.04em;
    box-shadow: 0 10px 30px rgba(79,70,229,0.35);
}

.brand-title {
    margin: 0;
    color: white;
    font-size: 1.15rem;
    font-weight: 760;
}

.brand-subtitle {
    margin: 0.35rem 0 0;
    color: #aeb8d7;
    font-size: 0.82rem;
    line-height: 1.55;
}

.sidebar-status {
    margin-top: 1.2rem;
    padding: 0.9rem;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0.85rem;
    background: rgba(255,255,255,0.045);
}

.sidebar-status-title {
    margin-bottom: 0.65rem;
    color: #aeb8d7;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0;
    font-size: 0.83rem;
}

.status-dot {
    display: inline-block;
    width: 0.48rem;
    height: 0.48rem;
    margin-right: 0.45rem;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 0 4px rgba(52,211,153,0.1);
}

.page-hero {
    position: relative;
    overflow: hidden;
    margin-bottom: 1.4rem;
    padding: 1.75rem 2rem;
    border: 1px solid rgba(79,70,229,0.12);
    border-radius: 1.2rem;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 16px 42px rgba(23,32,51,0.06);
}

.workspace-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--line);
}

.workspace-heading {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.workspace-icon,
.section-icon,
.panel-icon,
.compact-icon,
.stat-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: var(--primary);
}

.workspace-icon {
    width: 2.7rem;
    height: 2.7rem;
    border: 1px solid #d7d9ff;
    border-radius: 0.85rem;
    background: linear-gradient(145deg, #f6f5ff, var(--primary-soft));
}

.workspace-icon svg {
    width: 1.35rem;
    height: 1.35rem;
}

.workspace-icon svg,
.section-icon svg,
.panel-icon svg,
.compact-icon svg,
.stat-icon svg {
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.workspace-title {
    margin: 0;
    color: var(--ink);
    font-size: 1.55rem;
    line-height: 1.25;
    letter-spacing: -0.025em;
}

.workspace-description {
    margin: 0.35rem 0 0;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

.workspace-state {
    display: inline-flex;
    align-items: center;
    flex-shrink: 0;
    padding: 0.38rem 0.65rem;
    border: 1px solid #cfd5e2;
    border-radius: 999px;
    background: white;
    color: #475467;
    font-size: 0.75rem;
    font-weight: 650;
}

.process-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.55rem;
    margin-bottom: 1rem;
}

.process-item {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.7rem 0.8rem;
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    background: rgba(255,255,255,0.78);
}

.process-item.active {
    border-color: #c7c9ff;
    background: var(--primary-soft);
}

.process-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.65rem;
    height: 1.65rem;
    flex-shrink: 0;
    border-radius: 50%;
    background: #e8ebf2;
    color: #667085;
    font-size: 0.72rem;
    font-weight: 760;
}

.process-item.active .process-number {
    background: var(--primary);
    color: white;
}

.process-copy strong {
    display: block;
    color: var(--ink);
    font-size: 0.8rem;
}

.process-copy span {
    display: block;
    margin-top: 0.08rem;
    color: var(--muted);
    font-size: 0.68rem;
}

.panel-title {
    margin: 0 0 0.2rem;
    color: var(--ink);
    font-size: 1rem;
}

.panel-help {
    margin: 0 0 0.8rem;
    color: var(--muted);
    font-size: 0.8rem;
}

.panel-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin-bottom: 0.8rem;
}

.panel-heading .panel-title,
.panel-heading .panel-help {
    margin: 0;
}

.panel-heading .panel-help {
    margin-top: 0.18rem;
}

.panel-icon {
    width: 2rem;
    height: 2rem;
    border-radius: 0.6rem;
    background: var(--primary-soft);
}

.panel-icon svg {
    width: 1rem;
    height: 1rem;
}

.compact-grid {
    display: grid;
    grid-template-columns: repeat(var(--compact-columns, 3), minmax(0, 1fr));
    gap: 0.55rem;
    margin: 0.7rem 0;
}

.compact-item {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    min-width: 0;
    padding: 0.65rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    background: rgba(255,255,255,0.82);
}

.compact-icon {
    width: 1.9rem;
    height: 1.9rem;
    border-radius: 0.55rem;
    background: var(--primary-soft);
}

.compact-icon svg {
    width: 0.95rem;
    height: 0.95rem;
}

.compact-copy {
    min-width: 0;
}

.compact-copy strong {
    display: block;
    overflow: hidden;
    color: var(--ink);
    font-size: 0.78rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.compact-copy span {
    display: block;
    margin-top: 0.08rem;
    overflow: hidden;
    color: var(--muted);
    font-size: 0.68rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.inline-notice {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    min-height: 2.9rem;
    padding: 0.55rem 0.75rem;
    border: 1px solid #d7d9ff;
    border-radius: 0.75rem;
    background: var(--primary-soft);
}

.inline-notice strong {
    display: block;
    color: var(--ink);
    font-size: 0.78rem;
}

.inline-notice span {
    display: block;
    margin-top: 0.05rem;
    color: var(--muted);
    font-size: 0.68rem;
}

.empty-state {
    padding: 2.5rem 1rem;
    border: 1px dashed #cfd5e2;
    border-radius: 0.9rem;
    background: rgba(255,255,255,0.58);
    text-align: center;
}

.empty-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.75rem;
    height: 2.75rem;
    margin-bottom: 0.75rem;
    border-radius: 0.9rem;
    background: var(--primary-soft);
    color: var(--primary);
}

.empty-icon svg {
    width: 1.3rem;
    height: 1.3rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.empty-state strong {
    display: block;
    color: var(--ink);
    font-size: 0.95rem;
}

.empty-state span {
    display: block;
    max-width: 34rem;
    margin: 0.4rem auto 0;
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.6;
}

.page-hero::after {
    content: "";
    position: absolute;
    right: -5rem;
    top: -6rem;
    width: 16rem;
    height: 16rem;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.18), transparent 68%);
}

.page-kicker {
    margin-bottom: 0.45rem;
    color: var(--primary);
    font-size: 0.76rem;
    font-weight: 760;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.page-title {
    position: relative;
    z-index: 1;
    margin: 0;
    color: var(--ink);
    font-size: clamp(1.8rem, 3vw, 2.65rem);
    line-height: 1.16;
    letter-spacing: -0.035em;
}

.page-description {
    position: relative;
    z-index: 1;
    max-width: 780px;
    margin: 0.7rem 0 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.75;
}

.section-heading {
    margin: 1.7rem 0 0.85rem;
}

.section-title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-icon {
    width: 1.65rem;
    height: 1.65rem;
    border-radius: 0.5rem;
    background: var(--primary-soft);
}

.section-icon svg {
    width: 0.9rem;
    height: 0.9rem;
}

.section-heading h3 {
    margin: 0;
    color: var(--ink);
    font-size: 1.15rem;
}

.section-heading p {
    margin: 0.3rem 0 0;
    color: var(--muted);
    font-size: 0.88rem;
}

.stat-card {
    min-height: 7.6rem;
    padding: 1rem 1.05rem;
    border: 1px solid var(--line);
    border-radius: 1rem;
    background: rgba(255,255,255,0.93);
    box-shadow: 0 8px 26px rgba(23,32,51,0.045);
}

.stat-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}

.stat-icon {
    width: 1.8rem;
    height: 1.8rem;
    border-radius: 0.55rem;
    background: var(--primary-soft);
}

.stat-icon svg {
    width: 0.95rem;
    height: 0.95rem;
}

.stat-card.positive .stat-icon { color: var(--positive); background: var(--positive-soft); }
.stat-card.neutral .stat-icon { color: var(--neutral); background: var(--neutral-soft); }
.stat-card.negative .stat-icon { color: var(--negative); background: var(--negative-soft); }

.stat-card.primary { border-top: 3px solid var(--primary); }
.stat-card.positive { border-top: 3px solid var(--positive); }
.stat-card.neutral { border-top: 3px solid var(--neutral); }
.stat-card.negative { border-top: 3px solid var(--negative); }

.stat-label {
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 650;
}

.stat-value {
    margin-top: 0.3rem;
    color: var(--ink);
    font-size: 1.75rem;
    font-weight: 780;
    letter-spacing: -0.035em;
}

.stat-detail {
    margin-top: 0.25rem;
    color: #8992a6;
    font-size: 0.77rem;
}

.feature-card {
    height: 100%;
    min-height: 10.5rem;
    padding: 1.2rem;
    border: 1px solid var(--line);
    border-radius: 1rem;
    background: rgba(255,255,255,0.92);
    box-shadow: 0 8px 28px rgba(23,32,51,0.04);
}

.feature-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.6rem;
    background: var(--primary-soft);
    color: var(--primary);
    font-size: 0.78rem;
    font-weight: 800;
}

.feature-card h4 {
    margin: 0.9rem 0 0.4rem;
    color: var(--ink);
    font-size: 1rem;
}

.feature-card p {
    margin: 0;
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.65;
}

.flow-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.65rem;
}

.flow-step {
    position: relative;
    padding: 0.9rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: 0.85rem;
    background: white;
    text-align: center;
}

.flow-step strong {
    display: block;
    color: var(--ink);
    font-size: 0.82rem;
}

.flow-step span {
    display: block;
    margin-top: 0.2rem;
    color: var(--muted);
    font-size: 0.7rem;
}

.result-hero {
    margin: 1.1rem 0;
    padding: 1.4rem 1.5rem;
    border: 1px solid var(--line);
    border-left-width: 5px;
    border-radius: 1rem;
    background: white;
    box-shadow: 0 10px 30px rgba(23,32,51,0.055);
}

.result-hero.positive { border-left-color: var(--positive); background: linear-gradient(100deg, var(--positive-soft), white 58%); }
.result-hero.neutral { border-left-color: var(--neutral); background: linear-gradient(100deg, var(--neutral-soft), white 58%); }
.result-hero.negative { border-left-color: var(--negative); background: linear-gradient(100deg, var(--negative-soft), white 58%); }

.result-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.result-summary {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.result-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.8rem;
    height: 2.8rem;
    flex-shrink: 0;
    border-radius: 0.9rem;
    background: rgba(255,255,255,0.72);
}

.result-icon svg {
    width: 1.35rem;
    height: 1.35rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.result-hero.positive .result-icon { color: var(--positive); }
.result-hero.neutral .result-icon { color: var(--neutral); }
.result-hero.negative .result-icon { color: var(--negative); }

.result-label {
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 780;
}

.result-desc {
    margin-top: 0.35rem;
    color: var(--muted);
    line-height: 1.6;
}

.confidence-box {
    min-width: 9rem;
    text-align: right;
}

.confidence-value {
    color: var(--ink);
    font-size: 1.5rem;
    font-weight: 780;
}

.confidence-label {
    color: var(--muted);
    font-size: 0.75rem;
}

.progress-track {
    height: 0.48rem;
    margin-top: 0.55rem;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(100,116,139,0.15);
}

.progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #818cf8, #4f46e5);
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    min-height: 2rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    padding: 0.38rem 0.7rem;
    border: 1px solid #dfe3ee;
    border-radius: 999px;
    background: white;
    color: #475467;
    font-size: 0.79rem;
    font-weight: 620;
}

.chip.primary {
    border-color: #cfd3ff;
    background: var(--primary-soft);
    color: #4338ca;
}

.chip.positive {
    border-color: #bdebd8;
    background: var(--positive-soft);
    color: var(--positive);
}

.chip.negative {
    border-color: #ffd0d4;
    background: var(--negative-soft);
    color: var(--negative);
}

.quote-card {
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 0.9rem;
    background: #fafbfe;
    color: #475467;
    line-height: 1.7;
}

.evidence-card,
.advice-card,
.similar-card {
    margin-bottom: 0.75rem;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 0.9rem;
    background: white;
}

.evidence-title,
.advice-title {
    color: var(--ink);
    font-weight: 730;
}

.evidence-text,
.advice-text {
    margin-top: 0.35rem;
    color: #475467;
    line-height: 1.65;
}

.evidence-source {
    margin-top: 0.55rem;
    color: #8992a6;
    font-size: 0.78rem;
}

.summary-card {
    padding: 1.2rem 1.3rem;
    border: 1px solid #d7d9ff;
    border-radius: 1rem;
    background: linear-gradient(120deg, #f3f1ff, #ffffff);
}

.summary-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-bottom: 0.7rem;
}

.summary-card p {
    margin: 0;
    color: #344054;
    font-size: 1rem;
    line-height: 1.75;
}

.badge {
    display: inline-flex;
    padding: 0.28rem 0.55rem;
    border-radius: 999px;
    background: #eef2ff;
    color: #4338ca;
    font-size: 0.72rem;
    font-weight: 720;
}

.badge.positive { background: var(--positive-soft); color: var(--positive); }
.badge.neutral { background: var(--neutral-soft); color: var(--neutral); }
.badge.negative { background: var(--negative-soft); color: var(--negative); }

.bar-list {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
}

.bar-row-top {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.35rem;
    color: #475467;
    font-size: 0.82rem;
    font-weight: 650;
}

.bar-track {
    height: 0.65rem;
    overflow: hidden;
    border-radius: 999px;
    background: #edf0f5;
}

.bar-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #818cf8, #4f46e5);
}

.bar-fill.positive { background: linear-gradient(90deg, #55c69a, #16855b); }
.bar-fill.neutral { background: linear-gradient(90deg, #f1b94c, #b56b08); }
.bar-fill.negative { background: linear-gradient(90deg, #ef7b86, #c43d4b); }

.callout {
    padding: 0.95rem 1.05rem;
    border: 1px solid #f0d8a7;
    border-radius: 0.85rem;
    background: #fff9ea;
    color: #7a4d06;
    font-size: 0.86rem;
    line-height: 1.65;
}

@media (max-width: 900px) {
    .flow-row { grid-template-columns: repeat(2, 1fr); }
    .process-strip { grid-template-columns: 1fr; }
    .compact-grid { grid-template-columns: 1fr; }
    .workspace-header { flex-direction: column; }
    .result-topline { align-items: flex-start; flex-direction: column; }
    .confidence-box { text-align: left; width: 100%; }
    section[data-testid="stSidebar"] { width: 250px !important; min-width: 250px !important; }
}
</style>
"""

PLATFORM_STYLES = """
<style>
:root {
    --platform-blue: #075dad;
    --platform-blue-dark: #064f96;
    --platform-blue-soft: #eaf3ff;
    --platform-bg: #f3f6fa;
    --platform-sidebar: #f5f7fa;
    --platform-border: #e2e8f0;
    --platform-text: #24364b;
    --platform-muted: #7b8a9e;
    --positive: #21a67a;
    --positive-soft: #eaf8f3;
    --neutral: #e39a2d;
    --neutral-soft: #fff6e6;
    --negative: #db5b68;
    --negative-soft: #fff0f2;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}

[data-testid="stHeader"] {
    display: none;
}

[data-testid="stAppViewContainer"] {
    background: var(--platform-bg);
    color: var(--platform-text);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1420px;
    padding: 5.25rem 1.4rem 2.5rem;
}

.platform-topbar {
    position: fixed;
    z-index: 999999;
    top: 0;
    left: 0;
    right: 0;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.4rem 0 0;
    background: var(--platform-blue);
    color: white;
    box-shadow: 0 1px 5px rgba(15, 54, 95, 0.2);
}

.platform-brand {
    height: 64px;
    display: flex;
    align-items: center;
}

.platform-menu {
    width: 64px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--platform-blue-dark);
    font-size: 1.45rem;
    font-weight: 700;
}

.platform-seal {
    width: 36px;
    height: 36px;
    margin-left: 1.25rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 2px solid rgba(255,255,255,0.82);
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 800;
}

.platform-name {
    margin-left: 0.75rem;
    font-size: 1.18rem;
    font-weight: 650;
    letter-spacing: 0.04em;
}

.platform-actions {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.platform-search {
    width: 280px;
    padding: 0.62rem 0.85rem;
    border-radius: 0.35rem;
    background: rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.76);
    font-size: 0.82rem;
}

.platform-user {
    padding: 0.48rem 0.78rem;
    border-radius: 0.35rem;
    background: rgba(255,255,255,0.1);
    font-size: 0.82rem;
    font-weight: 650;
}

section[data-testid="stSidebar"] {
    width: 250px !important;
    min-width: 250px !important;
    padding-top: 64px;
    background: var(--platform-sidebar);
    border-right: 1px solid #dce3eb;
}

section[data-testid="stSidebar"] > div {
    padding: 1rem 0.75rem;
}

section[data-testid="stSidebar"] * {
    color: var(--platform-text);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.35rem;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    min-height: 2.85rem;
    padding: 0.62rem 0.8rem;
    border: 1px solid transparent;
    border-radius: 0.35rem;
    background: transparent;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #e9f1fb;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    border-color: #c5ddfa;
    background: #dfeeff;
    box-shadow: inset 3px 0 0 var(--platform-blue);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: var(--platform-blue);
    font-weight: 700;
}

.sidebar-product {
    padding: 0.45rem 0.65rem 0.95rem;
    border-bottom: 1px solid #dde4ec;
    margin-bottom: 0.75rem;
}

.sidebar-product strong {
    display: block;
    color: #1e3856;
    font-size: 0.92rem;
}

.sidebar-product span {
    display: block;
    margin-top: 0.2rem;
    color: var(--platform-muted);
    font-size: 0.72rem;
}

.sidebar-status {
    margin-top: 1.2rem;
    padding: 0.8rem;
    border: 1px solid #dfe6ee;
    border-radius: 0.4rem;
    background: white;
}

.sidebar-status-title {
    margin-bottom: 0.55rem;
    color: var(--platform-muted);
    font-size: 0.7rem;
    font-weight: 700;
}

.status-row {
    display: flex;
    justify-content: space-between;
    padding: 0.22rem 0;
    font-size: 0.76rem;
}

.status-dot {
    display: inline-block;
    width: 0.42rem;
    height: 0.42rem;
    margin-right: 0.4rem;
    border-radius: 50%;
    background: #20b486;
}

.workspace-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.9rem;
    padding: 0.9rem 1rem;
    border: 1px solid var(--platform-border);
    border-top: 3px solid var(--platform-blue);
    background: white;
}

.workspace-title {
    margin: 0;
    color: #1f3651;
    font-size: 1.32rem;
    font-weight: 700;
}

.workspace-description {
    margin: 0.25rem 0 0;
    color: var(--platform-muted);
    font-size: 0.8rem;
}

.workspace-state {
    padding: 0.32rem 0.65rem;
    border: 1px solid #bcd8f7;
    border-radius: 0.25rem;
    background: var(--platform-blue-soft);
    color: var(--platform-blue);
    font-size: 0.72rem;
    font-weight: 700;
}

.process-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    margin-bottom: 0.9rem;
    border: 1px solid var(--platform-border);
    background: white;
}

.process-item {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.65rem 0.8rem;
    border-right: 1px solid var(--platform-border);
}

.process-item:last-child {
    border-right: 0;
}

.process-item.active {
    background: var(--platform-blue-soft);
}

.process-number {
    width: 1.55rem;
    height: 1.55rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #dce4ed;
    color: #708198;
    font-size: 0.68rem;
    font-weight: 750;
}

.process-item.active .process-number {
    background: var(--platform-blue);
    color: white;
}

.process-copy strong {
    display: block;
    color: var(--platform-text);
    font-size: 0.77rem;
}

.process-copy span {
    display: block;
    margin-top: 0.08rem;
    color: var(--platform-muted);
    font-size: 0.65rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--platform-border) !important;
    border-radius: 0.4rem !important;
    background: white;
    box-shadow: 0 1px 3px rgba(32, 63, 94, 0.04);
}

.panel-title,
.section-heading h3 {
    margin: 0;
    color: #203851;
    font-size: 0.98rem;
}

.panel-help,
.section-heading p {
    margin: 0.2rem 0 0.7rem;
    color: var(--platform-muted);
    font-size: 0.76rem;
}

.section-heading {
    margin: 1.1rem 0 0.65rem;
    padding-left: 0.65rem;
    border-left: 3px solid var(--platform-blue);
}

[data-testid="stTextArea"] textarea,
[data-testid="stFileUploaderDropzone"],
[data-baseweb="select"] > div {
    border-radius: 0.3rem !important;
    border-color: #cfd9e5 !important;
    background: white !important;
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stDownloadButton > button {
    min-height: 2.5rem;
    border-radius: 0.3rem;
    font-weight: 650;
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    border-color: var(--platform-blue);
    background: var(--platform-blue);
}

.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    background: var(--platform-blue-dark);
}

.stat-card {
    min-height: 6.4rem;
    padding: 0.85rem 0.95rem;
    border: 1px solid var(--platform-border);
    border-top: 3px solid var(--platform-blue);
    border-radius: 0.35rem;
    background: white;
}

.stat-card.positive { border-top-color: var(--positive); }
.stat-card.neutral { border-top-color: var(--neutral); }
.stat-card.negative { border-top-color: var(--negative); }
.stat-label { color: var(--platform-muted); font-size: 0.74rem; }
.stat-value { margin-top: 0.28rem; color: #203851; font-size: 1.45rem; font-weight: 750; }
.stat-detail { margin-top: 0.22rem; color: #94a0af; font-size: 0.7rem; }

.result-hero {
    margin: 0.9rem 0;
    padding: 1rem 1.15rem;
    border: 1px solid var(--platform-border);
    border-left: 4px solid var(--platform-blue);
    border-radius: 0.35rem;
    background: white;
}

.result-hero.positive { border-left-color: var(--positive); background: var(--positive-soft); }
.result-hero.neutral { border-left-color: var(--neutral); background: var(--neutral-soft); }
.result-hero.negative { border-left-color: var(--negative); background: var(--negative-soft); }
.result-topline { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.result-label { color: #203851; font-size: 1.2rem; font-weight: 750; }
.result-desc { margin-top: 0.25rem; color: #65778c; font-size: 0.8rem; }
.confidence-box { min-width: 8rem; text-align: right; }
.confidence-value { color: #203851; font-size: 1.25rem; font-weight: 750; }
.confidence-label { color: var(--platform-muted); font-size: 0.68rem; }

.progress-track,
.bar-track {
    height: 0.45rem;
    overflow: hidden;
    border-radius: 999px;
    background: #e5ebf2;
}

.progress-fill,
.bar-fill {
    height: 100%;
    background: var(--platform-blue);
}

.bar-fill.positive { background: var(--positive); }
.bar-fill.neutral { background: var(--neutral); }
.bar-fill.negative { background: var(--negative); }
.bar-list { display: flex; flex-direction: column; gap: 0.75rem; }
.bar-row-top { display: flex; justify-content: space-between; margin-bottom: 0.28rem; color: #53667b; font-size: 0.74rem; }

.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip {
    display: inline-flex;
    padding: 0.3rem 0.55rem;
    border: 1px solid #d8e1eb;
    border-radius: 0.25rem;
    background: #f8fafc;
    color: #53667b;
    font-size: 0.72rem;
}
.chip.primary { border-color: #c4dcf8; background: var(--platform-blue-soft); color: var(--platform-blue); }

.quote-card,
.evidence-card,
.advice-card,
.similar-card,
.summary-card {
    margin-bottom: 0.65rem;
    padding: 0.8rem 0.9rem;
    border: 1px solid var(--platform-border);
    border-radius: 0.35rem;
    background: white;
}

.summary-card { border-left: 3px solid var(--platform-blue); background: #f7fbff; }
.summary-card p,
.evidence-text,
.advice-text { margin: 0.25rem 0 0; color: #53667b; font-size: 0.8rem; line-height: 1.6; }
.evidence-title,
.advice-title { color: #203851; font-size: 0.82rem; font-weight: 700; }
.evidence-source { margin-top: 0.4rem; color: #8795a8; font-size: 0.7rem; }
.summary-meta { display: flex; gap: 0.4rem; margin-bottom: 0.5rem; }
.badge {
    display: inline-flex;
    padding: 0.22rem 0.45rem;
    border-radius: 0.2rem;
    background: var(--platform-blue-soft);
    color: var(--platform-blue);
    font-size: 0.66rem;
    font-weight: 700;
}
.badge.positive { background: var(--positive-soft); color: var(--positive); }
.badge.neutral { background: var(--neutral-soft); color: #a36b17; }
.badge.negative { background: var(--negative-soft); color: var(--negative); }

.empty-state {
    padding: 2rem 1rem;
    border: 1px dashed #cdd8e4;
    border-radius: 0.35rem;
    background: white;
    text-align: center;
}
.empty-state strong { display: block; color: #344b64; font-size: 0.88rem; }
.empty-state span { display: block; margin-top: 0.35rem; color: var(--platform-muted); font-size: 0.74rem; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--platform-border);
    background: white;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 2.5rem;
    border-radius: 0;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--platform-blue);
    box-shadow: inset 0 -2px 0 var(--platform-blue);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--platform-border);
    border-radius: 0.3rem;
    overflow: hidden;
}

@media (max-width: 900px) {
    .platform-search { display: none; }
    .platform-name { font-size: 0.95rem; }
    .process-strip { grid-template-columns: 1fr; }
    .process-item { border-right: 0; border-bottom: 1px solid var(--platform-border); }
    section[data-testid="stSidebar"] { width: 230px !important; min-width: 230px !important; }
    .result-topline { align-items: flex-start; flex-direction: column; }
    .confidence-box { width: 100%; text-align: left; }
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_reference_reviews() -> list[str]:
    if not SAMPLE_DATA.exists():
        return []
    return rows_to_texts(load_reviews_csv(SAMPLE_DATA))


@st.cache_data(show_spinner=False)
def load_sample_rows() -> list[dict[str, str]]:
    if not SAMPLE_DATA.exists():
        return []
    return load_reviews_csv(SAMPLE_DATA)


@st.cache_data(show_spinner=False)
def load_test_cases() -> pd.DataFrame:
    if not TEST_CASES.exists():
        return pd.DataFrame()
    return pd.read_csv(TEST_CASES)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    if not MODEL_METRICS.exists():
        return {}
    return json.loads(MODEL_METRICS.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_bert_metrics() -> dict:
    if not BERT_METRICS.exists():
        return {}
    return json.loads(BERT_METRICS.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_ablation_metrics() -> dict:
    if not ABLATION_METRICS.exists():
        return {}
    return json.loads(ABLATION_METRICS.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def overview_statistics(
    rows: tuple[tuple[str, str], ...],
) -> tuple[dict[str, int], dict[str, int]]:
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    topics: dict[str, int] = {}
    for text, label in rows:
        if label in sentiments:
            sentiments[label] += 1
        for topic in detect_topics(text):
            topics[topic] = topics.get(topic, 0) + 1
    ordered_topics = dict(
        sorted(topics.items(), key=lambda item: item[1], reverse=True)
    )
    return sentiments, ordered_topics


@st.cache_data(show_spinner=False, max_entries=12)
def analyze_texts_cached(
    texts: tuple[str, ...],
    model_version: tuple[int, int],
) -> list[dict]:
    del model_version
    return analyze_batch(list(texts), use_llm=False)


def current_model_version() -> tuple[int, int]:
    model_path = Path("models/sentiment_model.pkl")
    vectorizer_path = Path("models/tfidf_vectorizer.pkl")
    return (
        model_path.stat().st_mtime_ns if model_path.exists() else 0,
        vectorizer_path.stat().st_mtime_ns if vectorizer_path.exists() else 0,
    )


@st.cache_data(show_spinner=False)
def llm_is_configured(env_version: int) -> bool:
    del env_version
    return bool(LLMConfig.from_env().api_key)


def current_env_version() -> int:
    env_path = Path(".env")
    return env_path.stat().st_mtime_ns if env_path.exists() else 0


def escaped(value: object) -> str:
    return html.escape(str(value))


def icon_svg(name: str) -> str:
    return ICON_SVGS.get(name, ICON_SVGS["spark"])


def render_page_intro(kicker: str, title: str, description: str) -> None:
    st.html(
        f"""
        <section class="page-hero">
            <div class="page-kicker">{escaped(kicker)}</div>
            <h1 class="page-title">{escaped(title)}</h1>
            <p class="page-description">{escaped(description)}</p>
        </section>
        """
    )


def render_workspace_header(
    title: str,
    description: str,
    state: str = "系统就绪",
    icon: str = "analysis",
) -> None:
    st.html(
        f"""
        <header class="workspace-header">
            <div class="workspace-heading">
                <span class="workspace-icon">{icon_svg(icon)}</span>
                <div>
                    <h1 class="workspace-title">{escaped(title)}</h1>
                    <p class="workspace-description">{escaped(description)}</p>
                </div>
            </div>
            <span class="workspace-state">{escaped(state)}</span>
        </header>
        """
    )


def render_process_strip(
    steps: list[tuple[str, str]],
    active_step: int,
) -> None:
    items = []
    for index, (title, detail) in enumerate(steps, start=1):
        active_class = " active" if index == active_step else ""
        items.append(
            f"""
            <div class="process-item{active_class}">
                <span class="process-number">{index}</span>
                <span class="process-copy">
                    <strong>{escaped(title)}</strong>
                    <span>{escaped(detail)}</span>
                </span>
            </div>
            """
        )
    st.html(f'<div class="process-strip">{"".join(items)}</div>')


def render_empty_state(title: str, description: str, icon: str = "spark") -> None:
    st.html(
        f"""
        <div class="empty-state">
            <span class="empty-icon">{icon_svg(icon)}</span>
            <strong>{escaped(title)}</strong>
            <span>{escaped(description)}</span>
        </div>
        """
    )


def render_section_heading(
    title: str,
    description: str = "",
    icon: str | None = None,
) -> None:
    detail = f"<p>{escaped(description)}</p>" if description else ""
    title_icon = (
        f'<span class="section-icon">{icon_svg(icon)}</span>'
        if icon
        else ""
    )
    st.html(
        f"""
        <div class="section-heading">
            <div class="section-title-row">
                {title_icon}
                <h3>{escaped(title)}</h3>
            </div>
            {detail}
        </div>
        """
    )


def render_stat_card(
    label: str,
    value: object,
    detail: str,
    tone: str = "primary",
    icon: str | None = None,
) -> None:
    stat_icon = (
        f'<span class="stat-icon">{icon_svg(icon)}</span>'
        if icon
        else ""
    )
    st.html(
        f"""
        <div class="stat-card {escaped(tone)}">
            <div class="stat-card-head">
                <div class="stat-label">{escaped(label)}</div>
                {stat_icon}
            </div>
            <div class="stat-value">{escaped(value)}</div>
            <div class="stat-detail">{escaped(detail)}</div>
        </div>
        """
    )


def render_feature_card(index: str, title: str, description: str) -> None:
    st.html(
        f"""
        <div class="feature-card">
            <span class="feature-index">{escaped(index)}</span>
            <h4>{escaped(title)}</h4>
            <p>{escaped(description)}</p>
        </div>
        """
    )


def render_panel_header(title: str, description: str, icon: str) -> None:
    st.html(
        f"""
        <div class="panel-heading">
            <span class="panel-icon">{icon_svg(icon)}</span>
            <div>
                <h3 class="panel-title">{escaped(title)}</h3>
                <p class="panel-help">{escaped(description)}</p>
            </div>
        </div>
        """
    )


def render_compact_items(
    items: list[tuple[str, str, str]],
    columns: int = 3,
) -> None:
    cards = "".join(
        f"""
        <div class="compact-item">
            <span class="compact-icon">{icon_svg(icon)}</span>
            <span class="compact-copy">
                <strong>{escaped(title)}</strong>
                <span>{escaped(detail)}</span>
            </span>
        </div>
        """
        for icon, title, detail in items
    )
    st.html(
        f'<div class="compact-grid" style="--compact-columns:{max(1, columns)}">{cards}</div>'
    )


def render_inline_notice(title: str, detail: str, icon: str = "check") -> None:
    st.html(
        f"""
        <div class="inline-notice">
            <span class="compact-icon">{icon_svg(icon)}</span>
            <span class="compact-copy">
                <strong>{escaped(title)}</strong>
                <span>{escaped(detail)}</span>
            </span>
        </div>
        """
    )


def render_chips(items: list[str], tone: str = "") -> None:
    if not items:
        st.caption("暂无识别结果")
        return
    chip_class = f"chip {tone}".strip()
    chips = "".join(
        f'<span class="{chip_class}">{escaped(item)}</span>'
        for item in items
    )
    st.html(f'<div class="chip-row">{chips}</div>')


def render_distribution_bars(
    data: dict[str, int],
    labels: dict[str, str] | None = None,
    sentiment_colors: bool = False,
) -> None:
    if not data:
        st.info("暂无可展示的数据。")
        return

    max_value = max(data.values()) or 1
    rows = []
    for key, value in data.items():
        width = max(4, round(value / max_value * 100))
        label = labels.get(key, key) if labels else key
        tone = key if sentiment_colors and key in SENTIMENT_UI else ""
        rows.append(
            f"""
            <div class="bar-row">
                <div class="bar-row-top">
                    <span>{escaped(label)}</span>
                    <span>{value}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill {escaped(tone)}" style="width:{width}%"></div>
                </div>
            </div>
            """
        )
    st.html(f'<div class="bar-list">{"".join(rows)}</div>')


def sync_sample_review() -> None:
    sample_key = st.session_state.get("sample_review", "中文评价")
    st.session_state["review_text"] = SAMPLE_REVIEWS[sample_key]


def render_analysis_result(result: dict) -> None:
    sentiment = str(result.get("sentiment", "neutral"))
    sentiment_ui = SENTIMENT_UI.get(sentiment, SENTIMENT_UI["neutral"])
    confidence = float(result.get("confidence", 0))
    confidence_percent = max(0, min(100, round(confidence * 100)))
    sentiment_icon = {
        "positive": "check",
        "neutral": "chart",
        "negative": "analysis",
    }.get(sentiment, "chart")

    st.html(
        f"""
        <section class="result-hero {sentiment_ui['tone']}">
            <div class="result-topline">
                <div class="result-summary">
                    <span class="result-icon">{icon_svg(sentiment_icon)}</span>
                    <div>
                        <div class="result-label">{escaped(sentiment_ui['label'])}</div>
                        <div class="result-desc">{escaped(sentiment_ui['description'])}</div>
                    </div>
                </div>
                <div class="confidence-box">
                    <div class="confidence-value">{confidence_percent}%</div>
                    <div class="confidence-label">分类置信度</div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width:{confidence_percent}%"></div>
                    </div>
                </div>
            </div>
        </section>
        """
    )

    stat_a, stat_b, stat_c = st.columns(3)
    with stat_a:
        render_stat_card(
            "语言识别",
            LANGUAGE_LABELS.get(result.get("language", "unknown"), result.get("language", "unknown")),
            "支持中文、英文及中英混合文本",
            icon="language",
        )
    with stat_b:
        render_stat_card(
            "判定来源",
            SENTIMENT_SOURCE_LABELS.get(
                result.get("sentiment_source", ""),
                result.get("sentiment_source", ""),
            ),
            "模型预测与规则校正协同",
            sentiment_ui["tone"],
            "model",
        )
    with stat_c:
        render_stat_card(
            "课程维度",
            len(result.get("topics", [])),
            "从评价中识别出的关注方面",
            icon="topic",
        )

    tab_result, tab_advice, tab_similar = st.tabs(
        ["结构化分析", "总结与改进建议", "相似评价"]
    )

    with tab_result:
        left, right = st.columns([1, 1])
        with left:
            render_section_heading(
                "课程维度",
                "将评价映射到可处理的教学环节",
                "topic",
            )
            render_chips([str(item) for item in result.get("topics", [])], "primary")
        with right:
            render_section_heading(
                "关键词",
                "保留最能代表评价内容的词语",
                "keyword",
            )
            render_chips([str(item) for item in result.get("keywords", [])])

        with st.expander("查看原始评价与课程证据"):
            render_section_heading(
                "原始评价",
                "分析结论始终回到学生原文",
                "text",
            )
            st.html(f'<div class="quote-card">{escaped(result.get("text", ""))}</div>')

            render_section_heading(
                "课程维度证据",
                "命中词与对应原文",
                "topic",
            )
            evidence_items = result.get("topic_evidence", [])
            if evidence_items:
                for item in evidence_items:
                    keywords = "、".join(str(word) for word in item.get("keywords", []))
                    st.html(
                        f"""
                        <div class="evidence-card">
                            <div class="evidence-title">{escaped(item.get("aspect", "课程维度"))}</div>
                            <div class="evidence-text">{escaped(item.get("evidence", ""))}</div>
                            <div class="evidence-source">命中关键词：{escaped(keywords)}</div>
                        </div>
                        """
                    )
            else:
                st.info("当前评价没有命中预设课程维度。")

        with st.expander("查看文本预处理结果"):
            st.code(result.get("processed_text") or "无", language="text")

    with tab_advice:
        advice = result.get("llm_advice")
        if not advice:
            st.info("本次分析未启用总结建议。打开“生成总结建议”后重新分析即可查看。")
        else:
            source = str(advice.get("source", "local_fallback"))
            source_label = "ChatECNU API" if source == "llm_api" else "本地兜底"
            risk_label, risk_tone = RISK_LABELS.get(
                str(advice.get("risk_level", "middle")),
                RISK_LABELS["middle"],
            )
            st.html(
                f"""
                <div class="summary-card">
                    <div class="summary-meta">
                        <span class="badge">{escaped(source_label)}</span>
                        <span class="badge {escaped(risk_tone)}">{escaped(risk_label)}</span>
                    </div>
                    <p>{escaped(advice.get("summary", "暂无总结"))}</p>
                </div>
                """
            )

            problems = advice.get("problems") or []
            render_section_heading(
                "需要关注的问题",
                "问题与课程维度及原文证据对应",
                "analysis",
            )
            if problems:
                for problem in problems:
                    st.html(
                        f"""
                        <div class="advice-card">
                            <div class="advice-title">{escaped(problem.get("aspect", "课程体验"))}</div>
                            <div class="advice-text">{escaped(problem.get("description", ""))}</div>
                            <div class="evidence-source">证据：{escaped(problem.get("evidence", ""))}</div>
                        </div>
                        """
                    )
            else:
                st.success("当前评价没有发现明确问题，重点是保持已有优势。")

            render_section_heading(
                "可执行改进建议",
                "面向教师或课程管理者的下一步行动",
                "spark",
            )
            for suggestion in advice.get("suggestions") or []:
                st.html(
                    f"""
                    <div class="advice-card">
                        <div class="advice-title">{escaped(suggestion.get("aspect", "课程体验"))}</div>
                        <div class="advice-text">{escaped(suggestion.get("suggestion", ""))}</div>
                        <div class="evidence-source">依据：{escaped(suggestion.get("evidence", ""))}</div>
                    </div>
                    """
                )

            with st.expander("查看结构化 JSON"):
                st.json(advice)

    with tab_similar:
        similar_reviews = result.get("similar_reviews", [])
        render_section_heading(
            "相似评价检索",
            "使用相似度辅助发现共性反馈",
            "batch",
        )
        if similar_reviews:
            for index, item in enumerate(similar_reviews, start=1):
                score = float(item.get("score", 0))
                score_percent = max(0, min(100, round(score * 100)))
                st.html(
                    f"""
                    <div class="similar-card">
                        <div class="bar-row-top">
                            <span>相似评价 {index}</span>
                            <span>相似度 {score_percent}%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width:{score_percent}%"></div>
                        </div>
                        <div class="evidence-text">{escaped(item.get("text", ""))}</div>
                    </div>
                    """
                )
        else:
            st.info("当前样本库中没有检索到相似评价。")


def rows_to_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["text", "course", "teacher", "label"])
    return pd.DataFrame(rows)


def result_rows(results: list[dict]) -> list[dict[str, object]]:
    return [
        {
            "评价文本": item["text"],
            "语言": LANGUAGE_LABELS.get(item["language"], item["language"]),
            "情感": SENTIMENT_LABELS.get(item["sentiment"], item["sentiment"]),
            "置信度": item["confidence"],
            "情感来源": SENTIMENT_SOURCE_LABELS.get(item.get("sentiment_source", ""), item.get("sentiment_source", "")),
            "主题": "、".join(item["topics"]),
            "关键词": "、".join(item["keywords"]),
        }
        for item in results
    ]


def bert_summary_rows(metrics: dict) -> list[dict[str, object]]:
    if not metrics:
        return []
    return [
        {
            "模型": "BERT",
            "预训练模型": metrics.get("model_name", "未知"),
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Macro-F1": f"{metrics.get('macro_f1', 0):.4f}",
            "测试样本": metrics.get("test_size", 0),
        }
    ]


def ablation_summary_rows(metrics: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    status_labels = {
        "completed": "完成",
        "完成": "完成",
        "skipped": "跳过",
        "跳过": "跳过",
    }
    for name, values in metrics.get("experiments", {}).items():
        status = status_labels.get(values.get("status", ""), values.get("status", "未知"))
        rows.append(
            {
                "实验版本": name,
                "状态": status,
                "Accuracy": (
                    f"{values.get('accuracy', 0):.4f}"
                    if values.get("accuracy") is not None
                    else "-"
                ),
                "Macro-F1": (
                    f"{values.get('macro_f1', 0):.4f}"
                    if values.get("macro_f1") is not None
                    else "-"
                ),
                "通过": values.get("passed", 0),
                "失败": values.get("failed", 0),
                "跳过": values.get("skipped", 0),
            }
        )
    return rows


def parse_expected_topics(value: object) -> list[str]:
    """解析测试用例表中的分号分隔预期主题。"""

    if value is None or pd.isna(value):
        return []

    return [
        topic.strip()
        for topic in str(value).replace("；", ";").split(";")
        if topic.strip()
    ]


def build_test_case_results(
    cases: pd.DataFrame,
    reference_reviews: list[str] | None = None,
) -> pd.DataFrame:
    """运行正式测试用例并返回适合报告展示的结果表。"""

    outputs: list[dict[str, object]] = []
    for _, row in cases.iterrows():
        result = analyze_review(
            str(row.get("text", "")),
            reference_reviews=reference_reviews,
            use_llm=False,
        )
        expected_topics = parse_expected_topics(row.get("expected_topics", ""))
        actual_topics = result["topics"]
        expected_sentiment = str(row.get("expected_sentiment", "")).strip()
        sentiment_passed = not expected_sentiment or result["sentiment"] == expected_sentiment
        topic_passed = set(expected_topics).issubset(set(actual_topics)) if expected_topics else True

        outputs.append(
            {
                "编号": row.get("id", ""),
                "评价文本": row.get("text", ""),
                "语言": LANGUAGE_LABELS.get(result["language"], result["language"]),
                "预期情感": expected_sentiment,
                "实际情感": result["sentiment"],
                "情感是否通过": "通过" if sentiment_passed else "需检查",
                "预期主题": "、".join(expected_topics),
                "实际主题": "、".join(actual_topics),
                "主题是否通过": "通过" if topic_passed else "需检查",
                "关键词": "、".join(result["keywords"]),
                "置信度": result["confidence"],
                "情感来源": SENTIMENT_SOURCE_LABELS.get(
                    result.get("sentiment_source", ""),
                    result.get("sentiment_source", ""),
                ),
                "备注": row.get("note", ""),
                "是否通过": "通过" if sentiment_passed and topic_passed else "需检查",
            }
        )

    return pd.DataFrame(outputs)


def render_shell() -> str:
    st.set_page_config(
        page_title="CourseInsight",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.html(APP_STYLES)

    st.sidebar.html(
        """
        <div class="brand-card">
            <div class="brand-mark">CI</div>
            <h2 class="brand-title">CourseInsight</h2>
            <p class="brand-subtitle">教学反馈分析平台</p>
        </div>
        """
    )
    page = st.sidebar.radio(
        "功能导航",
        PAGE_OPTIONS,
        label_visibility="collapsed",
    )

    model_ready = (Path("models/sentiment_model.pkl").exists()
                   and Path("models/tfidf_vectorizer.pkl").exists())
    llm_ready = llm_is_configured(current_env_version())
    st.sidebar.html(
        f"""
        <div class="sidebar-status">
            <div class="sidebar-status-title">运行状态</div>
            <div class="status-row">
                <span><span class="status-dot"></span>情感模型</span>
                <span>{"已加载" if model_ready else "规则模式"}</span>
            </div>
            <div class="status-row">
                <span><span class="status-dot"></span>LLM 服务</span>
                <span>{"已配置" if llm_ready else "本地兜底"}</span>
            </div>
        </div>
        """
    )
    st.sidebar.caption("CourseInsight v1.0")
    return page


def render_overview_page() -> None:
    render_page_intro(
        "NLP Application",
        "让课程评价从文本变成可执行的改进线索",
        "系统面向高校教师、课程负责人和教学管理者，将中英文学生评价转化为情感趋势、"
        "课程维度证据、相似反馈与改进建议。",
    )

    rows = load_sample_rows()
    texts = rows_to_texts(rows)
    overview_rows = tuple(
        (str(row.get("text", "")), str(row.get("label", "")))
        for row in rows
    )
    sentiments, topics = overview_statistics(overview_rows)

    metrics = load_model_metrics()

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_stat_card("双语训练数据", "9,120", "英文 9,000 条，中文 120 条")
    with col_b:
        render_stat_card(
            "最佳模型",
            metrics.get("best_model", "Logistic Regression"),
            "4 种传统分类器对比选优",
            "positive",
        )
    with col_c:
        render_stat_card(
            "Macro-F1",
            f"{metrics.get('macro_f1', 0):.3f}",
            "三分类整体均衡指标",
            "neutral",
        )
    with col_d:
        test_count = len(load_test_cases())
        render_stat_card("正式测试案例", test_count, "覆盖中英文本与混合情感")

    render_section_heading(
        "系统数据流",
        "从原始评价到教学建议，每一步都保留可解释的中间结果。",
    )
    st.html(
        """
        <div class="flow-row">
            <div class="flow-step"><strong>评价输入</strong><span>单条文本 / CSV</span></div>
            <div class="flow-step"><strong>双语预处理</strong><span>清洗、分词、停用词</span></div>
            <div class="flow-step"><strong>情感分类</strong><span>TF-IDF + 传统模型</span></div>
            <div class="flow-step"><strong>维度提取</strong><span>主题、关键词、证据</span></div>
            <div class="flow-step"><strong>相似检索</strong><span>余弦相似度</span></div>
            <div class="flow-step"><strong>建议生成</strong><span>LLM + 本地兜底</span></div>
        </div>
        """
    )

    render_section_heading(
        "核心能力",
        "围绕课程评价实际需求设计，兼顾分析效果、可解释性与演示稳定性。",
    )
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        render_feature_card(
            "01",
            "中英双语理解",
            "支持中文、英文和中英混合文本，分别执行适配的分词与停用词处理。",
        )
    with col_2:
        render_feature_card(
            "02",
            "可解释情感分类",
            "模型预测结合规则校正，展示置信度、判定来源和混合评价处理过程。",
        )
    with col_3:
        render_feature_card(
            "03",
            "课程维度证据",
            "识别教学内容、授课方式、作业、考试、实验与学习收获，并绑定原文。",
        )
    with col_4:
        render_feature_card(
            "04",
            "改进建议生成",
            "基于结构化结果调用 ChatECNU；接口异常时自动切换本地模板。",
        )

    render_section_heading("内置样本概况", "快速查看项目自带课程评价的情感与主题分布。")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            st.markdown("##### 情感分布")
            render_distribution_bars(
                sentiments,
                labels={
                    "positive": "正面",
                    "neutral": "中性 / 混合",
                    "negative": "负面",
                },
                sentiment_colors=True,
            )
    with chart_right:
        with st.container(border=True):
            st.markdown("##### 课程维度分布")
            render_distribution_bars(dict(list(topics.items())[:6]))

    with st.expander("查看项目内置样本数据"):
        st.dataframe(rows_to_frame(rows).head(12), width="stretch", hide_index=True)


def render_single_review_page() -> None:
    result = st.session_state.get("single_result")
    render_workspace_header(
        "评价分析",
        "输入学生反馈，识别情感与课程问题，并生成改进建议。",
        "分析完成" if result else "等待输入",
        "analysis",
    )
    render_process_strip(
        [
            ("输入评价", "填写或选择示例"),
            ("运行分析", "NLP 与大模型处理"),
            ("查看结果", "结论、证据与建议"),
        ],
        3 if result else 1,
    )

    if "sample_review" not in st.session_state:
        st.session_state["sample_review"] = "中文评价"
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = SAMPLE_REVIEWS["中文评价"]

    input_col, setting_col = st.columns([2.25, 0.75])
    with input_col:
        with st.form("single_review_form", border=True):
            render_panel_header(
                "评价内容",
                "支持中文、英文及中英混合文本。",
                "text",
            )
            text = st.text_area(
                "评价文本",
                key="review_text",
                height=170,
                placeholder="请输入学生对课程、教师、作业或实验的评价...",
                label_visibility="collapsed",
            )
            analyze_clicked = st.form_submit_button(
                "开始分析",
                type="primary",
                width="stretch",
            )
    with setting_col:
        with st.container(border=True):
            render_panel_header(
                "分析设置",
                "选择示例和输出方式。",
                "settings",
            )
            st.selectbox(
                "示例评价",
                list(SAMPLE_REVIEWS),
                key="sample_review",
                on_change=sync_sample_review,
            )
            use_llm = st.toggle(
                "生成改进建议",
                value=True,
                help="调用 ChatECNU；接口异常时使用本地兜底。",
            )
            render_compact_items(
                [
                    ("analysis", "情感识别", "自动"),
                    ("topic", "课程维度", "自动"),
                    ("keyword", "关键词", "自动"),
                ],
                columns=1,
            )

    if analyze_clicked:
        if not text.strip():
            st.warning("请输入评价文本后再开始分析。")
        else:
            with st.spinner("正在分析评价，请稍候..."):
                result = analyze_review(
                    text,
                    reference_reviews=load_reference_reviews(),
                    use_llm=use_llm,
                )
                st.session_state["single_result"] = result
            st.rerun()

    result = st.session_state.get("single_result")
    if result:
        render_analysis_result(result)
    else:
        render_empty_state(
            "暂无分析结果",
            "在上方输入一条课程评价并点击“开始分析”，结果将在此处展示。",
            "analysis",
        )


def load_uploaded_or_sample_rows(uploaded_file) -> tuple[list[dict[str, str]], str]:
    if uploaded_file is None:
        return load_sample_rows(), "内置样本数据"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    try:
        return load_reviews_csv(temp_path), uploaded_file.name
    finally:
        Path(temp_path).unlink(missing_ok=True)


def render_batch_page() -> None:
    batch_has_result = bool(st.session_state.get("batch_results"))
    render_workspace_header(
        "批量分析",
        "导入课程评价 CSV，查看整体趋势并导出分析结果。",
        "分析完成" if batch_has_result else "等待导入",
        "batch",
    )
    render_process_strip(
        [
            ("导入数据", "上传 CSV 或使用样本"),
            ("执行分析", "计算情感与课程维度"),
            ("导出结果", "查看明细并下载"),
        ],
        3 if batch_has_result else 1,
    )

    import_col, guide_col = st.columns([2, 1])
    with import_col:
        with st.container(border=True):
            render_panel_header(
                "导入评价数据",
                "上传 CSV；未上传时使用内置样本。",
                "upload",
            )
            uploaded = st.file_uploader(
                "课程评论 CSV",
                type=["csv"],
                label_visibility="collapsed",
            )
    with guide_col:
        with st.container(border=True):
            render_panel_header(
                "数据要求",
                "自动识别常见字段。",
                "data",
            )
            render_compact_items(
                [
                    ("text", "文本字段", "text / review"),
                    ("chart", "评分字段", "rating 可选"),
                    ("language", "文件编码", "UTF-8"),
                ],
                columns=1,
            )
    try:
        rows, source_name = load_uploaded_or_sample_rows(uploaded)
    except Exception as exc:
        st.error(f"CSV 读取失败：{exc}")
        return

    texts = rows_to_texts(rows)
    if not texts:
        st.warning("没有可分析的文本。")
        return

    dataset_key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    has_current_results = (
        st.session_state.get("batch_dataset_key") == dataset_key
        and bool(st.session_state.get("batch_results"))
    )

    source_col, action_col = st.columns([2, 1])
    source_col.caption(f"当前数据：{source_name} · {len(texts)} 条有效评价")
    analyze_clicked = action_col.button(
        "重新分析当前数据" if has_current_results else "开始批量分析",
        type="primary",
        width="stretch",
    )

    if analyze_clicked:
        with st.spinner("正在分析批量评价..."):
            st.session_state["batch_results"] = analyze_texts_cached(
                tuple(texts),
                current_model_version(),
            )
            st.session_state["batch_dataset_key"] = dataset_key
        st.rerun()

    if not has_current_results:
        with st.expander("预览待分析数据", expanded=True):
            st.dataframe(
                rows_to_frame(rows).head(20),
                width="stretch",
                hide_index=True,
            )
        render_empty_state(
            "数据已就绪",
            "确认数据内容后点击“开始批量分析”，系统将生成整体统计和逐条结果。",
            "data",
        )
        return

    results = st.session_state.get("batch_results")
    if not results:
        return

    sentiments = sentiment_distribution(results)
    topics = topic_distribution(results)
    keywords = dict(extract_keywords(texts, top_k=12))
    total = len(results)
    positive_rate = sentiments.get("positive", 0) / total if total else 0
    negative_rate = sentiments.get("negative", 0) / total if total else 0
    top_topic = next(iter(topics), "未识别")

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    with metric_a:
        render_stat_card("有效评价", total, f"来源：{source_name}", icon="data")
    with metric_b:
        render_stat_card(
            "正面占比",
            f"{positive_rate:.1%}",
            "课程认可度参考",
            "positive",
            "chart",
        )
    with metric_c:
        render_stat_card(
            "负面占比",
            f"{negative_rate:.1%}",
            "需优先关注的反馈",
            "negative",
            "analysis",
        )
    with metric_d:
        render_stat_card(
            "首要课程维度",
            top_topic,
            "出现频率最高的主题",
            "neutral",
            "topic",
        )

    render_section_heading("整体趋势", icon="chart")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        with st.container(border=True):
            st.markdown("##### 情感分布")
            render_distribution_bars(
                sentiments,
                labels={
                    "positive": "正面",
                    "neutral": "中性 / 混合",
                    "negative": "负面",
                },
                sentiment_colors=True,
            )
    with chart_b:
        with st.container(border=True):
            st.markdown("##### 课程维度分布")
            render_distribution_bars(dict(list(topics.items())[:8]))

    render_section_heading("高频关键词", icon="keyword")
    if keywords:
        weighted_keywords = [f"{word} · {count}" for word, count in keywords.items()]
        render_chips(weighted_keywords, "primary")
    else:
        st.info("暂未提取到关键词。")

    with st.expander("查看原始数据预览"):
        st.dataframe(
            rows_to_frame(rows).head(20),
            width="stretch",
            hide_index=True,
        )

    render_section_heading(
        "分析明细",
        "可筛查具体评价并导出结果。",
        "data",
    )
    results_df = pd.DataFrame(result_rows(results))
    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "评价文本": st.column_config.TextColumn("评价文本", width="large"),
            "置信度": st.column_config.ProgressColumn(
                "置信度",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
        },
    )
    st.download_button(
        "下载分析结果 CSV",
        data=results_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="courseinsight_batch_results.csv",
        mime="text/csv",
    )


def render_test_cases_page() -> None:
    output_df = st.session_state.get("test_case_results")
    render_workspace_header(
        "测试验证",
        "运行固定测试案例，检查情感分类和课程维度识别是否符合预期。",
        "验证完成" if output_df is not None else "等待执行",
        "check",
    )
    if not TEST_CASES.exists():
        st.warning("未找到测试用例文件。")
        return

    cases = load_test_cases()
    action_col, info_col = st.columns([1, 2])
    if action_col.button("运行全部测试", type="primary", width="stretch"):
        with st.spinner("正在运行测试案例..."):
            output_df = build_test_case_results(
                cases,
                reference_reviews=load_reference_reviews(),
            )
            st.session_state["test_case_results"] = output_df
    with info_col:
        render_inline_notice(
            f"{len(cases)} 个固定案例",
            "覆盖中文、英文和混合评价",
            "check",
        )

    if output_df is None:
        output_df = st.session_state.get("test_case_results")
    if output_df is None:
        with st.expander("查看测试案例", expanded=True):
            st.dataframe(cases, width="stretch", hide_index=True)
        return

    passed_count = int((output_df["是否通过"] == "通过").sum())
    pass_rate = passed_count / len(output_df) if len(output_df) else 0
    failed_count = len(output_df) - passed_count
    result_a, result_b, result_c = st.columns(3)
    with result_a:
        render_stat_card(
            "通过数量",
            passed_count,
            "情感与主题均符合预期",
            "positive",
            "check",
        )
    with result_b:
        render_stat_card(
            "需检查",
            failed_count,
            "建议人工复核的案例",
            "negative" if failed_count else "positive",
            "analysis",
        )
    with result_c:
        render_stat_card(
            "总通过率",
            f"{pass_rate:.1%}",
            "正式演示案例端到端结果",
            "positive" if pass_rate == 1 else "neutral",
            "chart",
        )

    if pass_rate == 1:
        st.success("全部测试案例均通过，当前版本满足演示用例要求。")
    else:
        st.warning("部分案例需要检查，请重点查看“需检查”行。")

    st.dataframe(
        output_df,
        width="stretch",
        hide_index=True,
        column_config={
            "评价文本": st.column_config.TextColumn("评价文本", width="large"),
            "置信度": st.column_config.ProgressColumn(
                "置信度",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
        },
    )
    st.download_button(
        "下载测试结果 CSV",
        data=output_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="courseinsight_test_case_results.csv",
        mime="text/csv",
    )


def render_tech_page() -> None:
    render_workspace_header(
        "系统信息",
        "查看模型状态、运行指标和系统组件。",
        "运行正常",
        "system",
    )

    metrics = load_model_metrics()
    if metrics:
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        with metric_a:
            render_stat_card(
                "最佳模型",
                metrics.get("best_model", "未知"),
                "按 Macro-F1 选择",
                icon="model",
            )
        with metric_b:
            render_stat_card(
                "Accuracy",
                f"{metrics.get('accuracy', 0):.4f}",
                "整体分类准确率",
                "positive",
                "check",
            )
        with metric_c:
            render_stat_card(
                "Macro-F1",
                f"{metrics.get('macro_f1', 0):.4f}",
                "三类等权平均",
                "neutral",
                "chart",
            )
        with metric_d:
            render_stat_card(
                "测试集",
                metrics.get("test_size", 0),
                "训练集外评估样本",
                icon="data",
            )

        render_section_heading("模型评估", icon="chart")
        rows = [
            {
                "模型": name,
                "准确率": f"{values['accuracy']:.4f}",
                "宏平均F1": f"{values['macro_f1']:.4f}",
            }
            for name, values in metrics.get("results", {}).items()
        ]
        table_col, language_col = st.columns([1.15, 0.85])
        with table_col:
            st.markdown("##### 分类器对比")
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )
        with language_col:
            st.markdown("##### 分语言评估")
            language_metrics = metrics.get("language_metrics", {})
            language_rows = []
            for language, values in language_metrics.items():
                language_rows.append(
                    {
                        "语言": LANGUAGE_LABELS.get(language, "总体" if language == "overall" else language),
                        "Accuracy": (
                            f"{values.get('accuracy', 0):.4f}"
                            if values.get("accuracy") is not None
                            else "-"
                        ),
                        "Macro-F1": (
                            f"{values.get('macro_f1', 0):.4f}"
                            if values.get("macro_f1") is not None
                            else "-"
                        ),
                        "样本": values.get("test_size", 0),
                    }
                )
            st.dataframe(
                pd.DataFrame(language_rows),
                width="stretch",
                hide_index=True,
            )

        with st.expander("查看模型图表与训练说明"):
            chart_a, chart_b = st.columns(2)
            with chart_a:
                if Path("outputs/charts/model_comparison.png").exists():
                    st.image(
                        "outputs/charts/model_comparison.png",
                        caption="模型指标对比",
                        width="stretch",
                    )
            with chart_b:
                if Path("outputs/charts/confusion_matrix.png").exists():
                    st.image(
                        "outputs/charts/confusion_matrix.png",
                        caption="混淆矩阵",
                        width="stretch",
                    )
            st.warning(
                "当前英文训练数据为 9,000 条，中文人工数据为 120 条；"
                "中文场景采用模型与规则联合校正。"
            )
    else:
        st.info("尚未生成模型指标。运行训练命令后会显示模型对比结果。")

    bert_rows = bert_summary_rows(load_bert_metrics())
    render_section_heading("BERT 对比实验", icon="model")
    if bert_rows:
        st.dataframe(pd.DataFrame(bert_rows), width="stretch", hide_index=True)
    else:
        st.info("尚未生成 BERT 指标。运行 BERT 对比实验后会显示结果。")

    ablation_rows = ablation_summary_rows(load_ablation_metrics())
    render_section_heading("消融实验", icon="check")
    if ablation_rows:
        st.dataframe(pd.DataFrame(ablation_rows), width="stretch", hide_index=True)
    else:
        st.info("尚未生成消融实验结果。运行消融脚本后会显示对比表。")

    with st.expander("查看系统组件"):
        render_compact_items(
            [
                ("data", "数据读取", "CSV 字段兼容"),
                ("text", "文本预处理", "双语清洗与分词"),
                ("model", "情感模型", "模型预测与校正"),
                ("topic", "课程维度", "主题与证据识别"),
                ("spark", "LLM 建议", "API 与本地兜底"),
            ],
            columns=3,
        )

    with st.expander("查看评分映射"):
        render_compact_items(
            [
                ("check", "4-5 分", "正面体验"),
                ("chart", "3 分", "中性或混合"),
                ("analysis", "1-2 分", "负面体验"),
            ],
            columns=3,
        )


def main() -> None:
    page = render_shell()
    if page == "评价分析":
        render_single_review_page()
    elif page == "批量分析":
        render_batch_page()
    elif page == "测试验证":
        render_test_cases_page()
    else:
        render_tech_page()


if __name__ == "__main__":
    main()
