"""CourseInsight Streamlit front end."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loader import load_reviews_csv, rows_to_texts
from src.keyword_extractor import extract_keywords
from src.llm_client import LLMConfig
from src.nlp_analyzer import (
    analyze_batch,
    analyze_review,
    sentiment_distribution,
    sentiment_runtime_status,
    topic_distribution,
)
from src.topic_analyzer import TOPIC_KEYWORDS


SAMPLE_DATA = Path("data/sample_reviews.csv")
MODEL_METRICS = Path("models/model_metrics.json")
BERT_METRICS = Path("outputs/bert_metrics.json")
ABLATION_METRICS = Path("outputs/reports/ablation_metrics.json")
CONFUSION_MATRIX_CHART = Path("outputs/charts/confusion_matrix.png")
BERT_CONFUSION_MATRIX_CHART = Path("outputs/charts/bert_confusion_matrix.png")
APP_LOGO = "🎓"

PAGE_OPTIONS = [
    "🏠 首页概览",
    "📄 单条分析",
    "📊 批量分析",
    "🧠 模型评估",
]

SAMPLE_REVIEWS = {
    "中文评价": "这门课老师讲解很清楚，案例也很贴近实际，课堂互动让我更容易理解重点。",
    "英文评价": "The lectures are organized, but the assignments are too heavy and feedback often comes late.",
    "中英混合": "老师讲得很认真，examples are helpful，但是实验部分 guidance 不够详细，希望能多给步骤说明。",
}
TEXT_COLUMN_CANDIDATES = ("review_text", "text", "review", "reviews", "comment", "content", "评价", "评论")
COURSE_COLUMN_CANDIDATES = ("course_name", "course", "course_title", "课程", "课程名称")
TEACHER_COLUMN_CANDIDATES = ("teacher", "instructor", "讲师", "教师")
RATING_COLUMN_CANDIDATES = ("rating", "stars", "score", "评分")
DATE_COLUMN_CANDIDATES = ("date", "created_at", "time", "日期", "时间")

SENTIMENT_LABELS = {
    "positive": "积极",
    "neutral": "中性",
    "negative": "消极",
}
SENTIMENT_DETAILS = {
    "positive": "整体反馈偏正向，可继续保留优势做法。",
    "neutral": "反馈包含认可与改进点，建议分项查看。",
    "negative": "反馈指出明确问题，建议优先复核证据。",
}
SENTIMENT_TONES = {
    "positive": "success",
    "neutral": "warning",
    "negative": "danger",
}
LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英文",
    "mixed": "中英混合",
    "unknown": "未知",
}
SOURCE_LABELS = {
    "model": "传统模型预测",
    "rule": "规则校正",
    "hybrid": "模型 + 规则校正",
    "bert": "BERT 模型预测",
    "bert+rule": "BERT 模型 + 规则校正",
    "tfidf": "传统文本模型",
    "tfidf+rule": "TF-IDF 模型 + 规则校正",
}
RISK_LABELS = {
    "low": ("低风险", "success"),
    "middle": ("中风险", "warning"),
    "high": ("高风险", "danger"),
}
MODEL_NAME_LABELS = {
    "Dummy Most Frequent": "Dummy Most Frequent",
    "LogisticRegression": "Logistic Regression",
    "logistic_regression": "Logistic Regression",
    "Logistic Regression": "Logistic Regression",
    "NaiveBayes": "Naive Bayes",
    "naive_bayes": "Naive Bayes",
    "Naive Bayes": "Naive Bayes",
    "LinearSVM": "Linear SVM",
    "linear_svm": "Linear SVM",
    "Linear SVM": "Linear SVM",
    "SVM": "SVM",
    "BERT": "BERT",
}
CHART_COLORS = {
    "primary": "#2563EB",
    "accent": "#F97316",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "muted": "#64748B",
    "compare": "#94A3B8",
    "accent_2": "#7C3AED",
}

APP_CSS = """
<style>
:root {
    --primary: #2563EB;
    --primary-2: #1D4ED8;
    --accent: #F97316;
    --accent-2: #7C3AED;
    --primary-soft: #EFF6FF;
    --accent-soft: #FFF7ED;
    --background: #F7F8FA;
    --card-bg: #FFFFFF;
    --text-main: #111827;
    --text-muted: #64748B;
    --border: #E5E7EB;
    --success: #16A34A;
    --warning: #F59E0B;
    --danger: #DC2626;
    --sidebar-bg: #111827;
    --card-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    --radius-card: 18px;
    --radius-control: 12px;
    --radius-inner: 14px;
}

html,
body,
[class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    color: var(--text-main);
    letter-spacing: 0;
}

#MainMenu,
footer,
[data-testid="stDecoration"] {
    display: none;
    visibility: hidden;
}

[data-testid="stHeader"] {
    height: 2rem;
    background: transparent;
}

[data-testid="stAppViewContainer"] {
    background: var(--background);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding: 2rem 1.5rem 3rem;
}

section[data-testid="stSidebar"] {
    width: 245px !important;
    min-width: 245px !important;
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, #0F172A 100%);
    border-right: 0;
}

section[data-testid="stSidebar"] > div {
    padding: 1.25rem 0.95rem;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: rgba(255, 255, 255, 0.78);
}

.sidebar-brand {
    padding: 0.25rem 0.35rem 1.05rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.sidebar-brand-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0;
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 700;
    line-height: 1.35;
}

.sidebar-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: var(--radius-control);
    background: rgba(255, 255, 255, 0.12);
    color: #FFFFFF;
    font-size: 18px;
    line-height: 1;
}

.sidebar-version {
    margin-top: 1rem;
    padding: 0.5rem 0.35rem;
    color: rgba(255, 255, 255, 0.56);
    font-size: 13px;
    line-height: 1.5;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
section[data-testid="stSidebar"] [data-testid="stRadio"] svg {
    display: none !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {
    padding-left: 0 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] p {
    margin-left: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    width: 100%;
    gap: 0.35rem;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center;
    width: 100% !important;
    box-sizing: border-box;
    min-height: 42px;
    padding: 0.72rem 0.95rem !important;
    border: 1px solid transparent;
    border-radius: var(--radius-control);
    background: transparent;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    width: 100%;
    margin: 0 !important;
    color: rgba(255, 255, 255, 0.78);
    font-size: 14px;
    line-height: 1.3;
    font-weight: 600;
    text-align: left;
    white-space: nowrap;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
    border-color: rgba(255, 255, 255, 0.18);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.24);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #FFFFFF;
    font-weight: 700;
}

.page-header {
    margin-bottom: 20px;
}

.page-title {
    margin: 0;
    color: var(--text-main);
    font-size: 28px;
    line-height: 1.28;
    font-weight: 700;
}

.page-subtitle {
    max-width: 840px;
    margin: 0.45rem 0 0;
    color: var(--text-muted);
    font-size: 15px;
    line-height: 1.7;
    font-weight: 400;
}

.hero-card,
.metric-card,
.info-card,
.flow-step,
.hero-status-panel,
.hero-status-item,
.result-card,
.notice-card,
.mini-card,
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-card) !important;
    background: var(--card-bg) !important;
    box-shadow: var(--card-shadow) !important;
}

.hero-card {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
    gap: 24px;
    align-items: center;
    margin-bottom: 20px;
    padding: 24px 26px;
}

.hero-title {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0;
    color: var(--text-main);
    font-size: 30px;
    line-height: 1.25;
    font-weight: 700;
}

.hero-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.7rem;
    height: 2.7rem;
    border-radius: var(--radius-inner);
    background: var(--primary-soft);
    color: var(--primary);
    font-size: 24px;
    line-height: 1;
}

.hero-copy {
    max-width: 720px;
    margin: 0.75rem 0 0;
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.72;
}

.hero-status-panel {
    padding: 16px;
}

.hero-status-title {
    margin: 0 0 12px;
    color: var(--text-main);
    font-size: 16px;
    font-weight: 700;
}

.hero-status-list {
    display: grid;
    gap: 10px;
}

.hero-status-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    min-height: 44px;
    padding: 10px 12px;
    box-shadow: none !important;
}

.hero-status-item span:first-child {
    color: var(--text-muted);
    font-size: 13px;
}

.hero-status-item span:last-child {
    color: var(--text-main);
    font-size: 14px;
    font-weight: 700;
    white-space: nowrap;
}

.hero-status-item.accent span:last-child {
    color: var(--accent);
}

.metric-card {
    min-height: 132px;
    padding: 20px;
    box-sizing: border-box;
    overflow: hidden;
}

.metric-head {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    margin-top: 0.75rem;
    color: var(--text-main);
    font-size: 34px;
    line-height: 1.15;
    font-weight: 700;
    white-space: nowrap;
    letter-spacing: 0;
}

.metric-detail {
    margin-top: 0.55rem;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.55;
    font-weight: 400;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.home-metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    align-items: stretch;
}

.home-metric-grid .metric-card {
    min-height: 148px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.home-metric-grid .metric-value {
    font-size: 34px;
}

.home-info-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    align-items: stretch;
}

.single-overview-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    align-items: stretch;
}

.single-overview-grid .metric-card {
    min-height: 142px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.section-title {
    margin: 24px 0 12px;
    color: var(--text-main);
    font-size: 20px;
    font-weight: 700;
}

.card-heading {
    margin: 0 0 0.35rem;
    color: var(--text-main);
    font-size: 16px;
    font-weight: 700;
}

.card-help {
    margin: 0 0 1rem;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.6;
}

.info-card,
.mini-card,
.analysis-summary-card,
.detail-card,
.advice-card,
.error-card {
    height: 100%;
    padding: 20px;
}

.info-card h3,
.mini-card h3,
.analysis-summary-card h3,
.detail-card h3,
.advice-card h3,
.error-card h3 {
    margin: 0 0 0.55rem;
    color: var(--text-main);
    font-size: 16px;
    font-weight: 700;
}

.info-card p,
.mini-card p,
.analysis-summary-card p,
.detail-card p,
.advice-card p,
.error-card p {
    margin: 0;
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.65;
}

.analysis-summary-card,
.detail-card,
.advice-card,
.error-card {
    border: 1px solid var(--border);
    border-radius: var(--radius-card);
    background: var(--card-bg);
    box-shadow: var(--card-shadow);
    box-sizing: border-box;
}

.analysis-summary-card {
    margin-bottom: 18px;
}

.analysis-summary-text {
    margin-bottom: 14px;
    color: var(--text-main);
    font-size: 15px;
    line-height: 1.7;
}

.analysis-summary-grid,
.detail-card-grid,
.advice-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    align-items: stretch;
}

.analysis-summary-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
}

.summary-stat {
    min-height: 64px;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-inner);
    background: var(--background);
    box-sizing: border-box;
}

.summary-stat span,
.detail-meta,
.advice-meta,
.error-meta {
    display: block;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.5;
}

.summary-stat strong,
.detail-value {
    display: block;
    margin-top: 0.25rem;
    color: var(--text-main);
    font-size: 15px;
    line-height: 1.45;
    font-weight: 700;
}

.detail-card,
.advice-card {
    min-height: 188px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.advice-card {
    min-height: 202px;
}

.advice-highlight {
    margin-bottom: 18px;
    border-left: 4px solid var(--accent);
    background: var(--accent-soft) !important;
}

.advice-highlight p {
    color: var(--text-main);
    font-size: 15px;
    line-height: 1.7;
}

.advice-highlight .advice-meta {
    color: var(--accent);
}

.detail-keywords {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.detail-keyword {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: var(--primary-soft);
    color: var(--primary);
    font-size: 13px;
    font-weight: 500;
}

.ui-badge {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 0.16rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--card-bg);
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.4;
    font-weight: 600;
}

.ui-badge.primary {
    border-color: rgba(37, 99, 235, 0.22);
    background: var(--primary-soft);
    color: var(--primary);
}

.ui-badge.accent {
    border-color: rgba(249, 115, 22, 0.25);
    background: var(--accent-soft);
    color: var(--accent);
}

.ui-badge.warning {
    border-color: rgba(245, 158, 11, 0.26);
    background: #FFFBEB;
    color: #B45309;
}

.ui-badge.success {
    border-color: rgba(22, 163, 74, 0.24);
    background: #F0FDF4;
    color: var(--success);
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}

.evidence-quote {
    margin-top: 8px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-control);
    background: var(--background);
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.6;
}

.error-card {
    border-left: 4px solid var(--danger);
}

.error-meta {
    margin-top: 8px;
}

.flow-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
    position: relative;
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-card);
    background: var(--card-bg);
    box-shadow: var(--card-shadow);
}

.flow-step {
    position: relative;
    min-height: 132px;
    padding: 16px 14px;
    box-shadow: none !important;
    background: var(--card-bg) !important;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.flow-step::after {
    content: "";
    position: absolute;
    top: 29px;
    right: -9px;
    width: 16px;
    height: 1px;
    background: rgba(37, 99, 235, 0.25);
}

.flow-step:last-child::after {
    display: none;
}

.flow-index {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 0 0 2.1rem;
    box-sizing: border-box;
    width: 2.1rem;
    height: 2.1rem;
    padding: 0;
    margin: 0;
    border-radius: 999px;
    background: var(--primary);
    color: #FFFFFF !important;
    font-size: 14px;
    font-weight: 800;
    line-height: 1 !important;
    text-align: center;
}

.flow-step strong {
    display: block;
    margin-top: 0.7rem;
    color: var(--text-main);
    font-size: 16px;
    font-weight: 700;
}

.flow-step > span:not(.flow-index) {
    display: block;
    margin-top: 0.28rem;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.45;
}

.tech-footnote {
    margin-top: 12px;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.6;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.chip {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 0.32rem 0.65rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--card-bg);
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 500;
}

.chip.primary {
    border-color: var(--primary);
    background: var(--primary-soft);
    color: var(--primary);
}

.chip.success {
    border-color: var(--success);
    color: var(--success);
}

.chip.warning {
    border-color: var(--warning);
    color: var(--text-main);
}

.chip.danger {
    border-color: var(--danger);
    color: var(--danger);
}

.result-card,
.notice-card {
    margin-bottom: 16px;
    padding: 18px 20px;
}

.result-card h3,
.notice-card h3 {
    margin: 0 0 0.45rem;
    color: var(--text-main);
    font-size: 16px;
    font-weight: 700;
}

.result-card p,
.notice-card p {
    margin: 0;
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.65;
}

.result-card.success {
    border-left: 4px solid var(--success) !important;
}

.result-card.warning {
    border-left: 4px solid var(--warning) !important;
}

.result-card.danger {
    border-left: 4px solid var(--danger) !important;
}

.progress-track {
    height: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--border);
}

.progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
}

.progress-fill.success {
    background: var(--success);
}

.progress-fill.warning {
    background: var(--warning);
}

.progress-fill.danger {
    background: var(--danger);
}

.evidence-meta {
    margin-top: 0.55rem;
    color: var(--text-muted);
    font-size: 13px;
}

.muted-text {
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.6;
}

[data-testid="stTextArea"] textarea,
[data-testid="stFileUploaderDropzone"],
[data-baseweb="select"] > div {
    border-color: var(--border) !important;
    border-radius: var(--radius-control) !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stFileUploaderDropzone"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stDownloadButton > button {
    min-height: 42px;
    border: 1px solid var(--border);
    border-radius: var(--radius-control);
    background: var(--card-bg);
    color: var(--text-main);
    font-size: 14px;
    font-weight: 600;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--primary);
    background: var(--primary-soft);
    color: var(--primary);
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    border-color: var(--primary);
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    color: #FFFFFF;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    border-color: var(--primary-2);
    background: linear-gradient(135deg, var(--primary-2), var(--primary));
    color: #FFFFFF;
}

.st-key-single_example_0 button,
.st-key-single_example_1 button,
.st-key-single_example_2 button {
    width: 100% !important;
    height: 44px !important;
    min-height: 44px !important;
    box-sizing: border-box;
    white-space: nowrap;
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-control);
}

@media (max-width: 1000px) {
    .home-metric-grid,
    .home-info-grid,
    .single-overview-grid,
    .analysis-summary-grid,
    .detail-card-grid,
    .advice-card-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .flow-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    section[data-testid="stSidebar"] {
        width: 245px !important;
        min-width: 245px !important;
    }

    .flow-grid {
        grid-template-columns: 1fr;
    }

    .home-metric-grid,
    .home-info-grid,
    .single-overview-grid,
    .analysis-summary-grid,
    .detail-card-grid,
    .advice-card-grid {
        grid-template-columns: 1fr;
    }

    .flow-step::after {
        display: none;
    }

    .hero-card {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def escaped(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def inject_css() -> None:
    st.html(APP_CSS)


@st.cache_resource(show_spinner=False)
def load_backend() -> dict[str, Any]:
    """Return real project backend entry points."""

    return {
        "single": analyze_review,
        "batch": analyze_batch,
        "runtime_status": sentiment_runtime_status,
    }


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


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
def llm_is_configured(env_version: int) -> bool:
    del env_version
    return bool(LLMConfig.from_env().api_key)


@st.cache_data(show_spinner=False, max_entries=12)
def run_batch_analysis_cached(
    texts: tuple[str, ...],
    model_version: tuple[int, ...],
) -> list[dict[str, Any]]:
    del model_version
    return load_backend()["batch"](list(texts), use_llm=False, reference_reviews=list(texts))


def run_single_analysis(text: str, use_llm: bool) -> dict[str, Any]:
    result = load_backend()["single"](
        text,
        reference_reviews=load_reference_reviews(),
        use_llm=use_llm,
    )
    validate_single_analysis_result(result)
    return result


def validate_single_analysis_result(result: object) -> None:
    if not isinstance(result, dict):
        raise TypeError(f"分析结果格式不符合预期，实际返回 {type(result).__name__}")

    required_fields = ("sentiment", "confidence", "keywords", "topic_evidence")
    missing = [field for field in required_fields if field not in result]
    if missing:
        available = ", ".join(sorted(str(key) for key in result.keys()))
        raise ValueError(f"分析结果缺少必要字段：{', '.join(missing)}。已返回字段：{available}")


def single_backend_error_payload(exc: Exception) -> dict[str, str]:
    return {
        "message": str(exc) or "分析过程没有返回更多信息",
    }


def current_env_version() -> int:
    env_path = Path(".env")
    return env_path.stat().st_mtime_ns if env_path.exists() else 0


def current_model_version() -> tuple[int, ...]:
    files = [
        Path("models/sentiment_model.pkl"),
        Path("models/tfidf_vectorizer.pkl"),
        Path("outputs/bert_model_final/config.json"),
        Path("outputs/bert_model_final/model.safetensors"),
        Path(".env"),
    ]
    return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in files)


def format_percent(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "缺少数据"


def format_decimal(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "缺少数据"


def count_with_unit(value: object, unit: str) -> str:
    try:
        return f"{int(value):,} {unit}"
    except (TypeError, ValueError):
        return f"{value} {unit}".strip()


def display_model_name(name: object) -> str:
    text = str(name or "").strip()
    if text in MODEL_NAME_LABELS:
        return MODEL_NAME_LABELS[text]
    normalized = text.replace("-", "_").replace(" ", "_").lower()
    if normalized in MODEL_NAME_LABELS:
        return MODEL_NAME_LABELS[normalized]
    if "logistic" in normalized:
        return "Logistic Regression"
    if "naive" in normalized or "bayes" in normalized:
        return "Naive Bayes"
    if "svm" in normalized:
        return "Linear SVM"
    if "bert" in normalized:
        return "BERT"
    return text or "缺少数据"


def active_model_short_label() -> str:
    status = load_backend()["runtime_status"]()
    active = status.get("active_backend", "rule")
    if active == "bert":
        return "BERT 情感分类 + 规则校正"
    if active == "tfidf":
        return "传统文本模型 + 规则校正"
    return "规则校正"


def training_sample_count() -> int:
    count, _detail = data_scale_summary()
    try:
        return int(str(count).replace(",", ""))
    except ValueError:
        return 0


def data_scale_summary() -> tuple[str, str]:
    bert_metrics = load_json_file(str(BERT_METRICS))
    total = sum(
        int(bert_metrics.get(key, 0) or 0)
        for key in ("train_size", "validation_size", "test_size")
    )
    if total:
        return f"{total:,}", "训练、验证与测试数据规模。"
    metrics = load_json_file(str(MODEL_METRICS))
    total = int(metrics.get("train_size", 0) or 0) + int(metrics.get("test_size", 0) or 0)
    if total:
        return f"{total:,}", "训练与测试数据规模。"
    sample_count = len(load_sample_rows())
    if sample_count:
        return f"{sample_count:,}", "项目示例数据规模。"
    return "缺少数据", "当前未找到可展示的数据规模。"


def active_model_card_value() -> tuple[str, str]:
    status = load_backend()["runtime_status"]()
    active = status.get("active_backend", "rule")
    if active == "bert":
        return "BERT", "结合规则校正完成情感分类。"
    if active == "tfidf":
        return "传统文本模型", "结合规则校正完成情感分类。"
    return "规则校正", "当前使用课程场景规则进行情感判断。"


def render_page_header(icon: str, title: str, subtitle: str) -> None:
    st.html(
        f"""
        <div class="page-header">
            <h1 class="page-title">{escaped(icon)} {escaped(title)}</h1>
            <p class="page-subtitle">{escaped(subtitle)}</p>
        </div>
        """
    )


def render_hero() -> None:
    status = load_backend()["runtime_status"]()
    active_model = active_model_short_label()
    device = status.get("bert", {}).get("device", "cpu") if status.get("active_backend") == "bert" else "cpu"
    advice_label = "已配置" if llm_is_configured(current_env_version()) else "未配置"
    st.html(
        f"""
        <section class="hero-card">
            <div>
                <h1 class="hero-title"><span class="hero-logo">{escaped(APP_LOGO)}</span>课程评价智能分析平台</h1>
                <p class="hero-copy">
                    面向课程评价文本，提供情感倾向识别、关键词抽取、问题维度定位与教学改进建议生成。
                </p>
            </div>
            <aside class="hero-status-panel">
                <h2 class="hero-status-title">系统状态</h2>
                <div class="hero-status-list">
                    <div class="hero-status-item"><span>当前模型</span><span>{escaped(active_model)}</span></div>
                    <div class="hero-status-item accent"><span>大模型配置</span><span>{escaped(advice_label)}</span></div>
                    <div class="hero-status-item"><span>推理设备</span><span>{escaped(str(device).upper())}</span></div>
                </div>
            </aside>
        </section>
        """
    )


def render_metric_card(
    icon: str,
    label: str,
    value: object,
    detail: str,
) -> None:
    st.html(metric_card_html(icon, label, value, detail))


def metric_card_html(icon: str, label: str, value: object, detail: str) -> str:
    icon_html = f"<span>{escaped(icon)}</span>" if str(icon).strip() else ""
    return f"""
        <div class="metric-card">
            <div class="metric-head">{icon_html}<span>{escaped(label)}</span></div>
            <div class="metric-value">{escaped(value)}</div>
            <div class="metric-detail">{escaped(detail)}</div>
        </div>
        """


def render_card_title(title: str, help_text: str = "") -> None:
    detail = f'<p class="card-help">{escaped(help_text)}</p>' if help_text else ""
    st.html(f'<h2 class="card-heading">{escaped(title)}</h2>{detail}')


def render_section_title(title: str) -> None:
    st.html(f'<h2 class="section-title">{escaped(title)}</h2>')


def render_info_card(title: str, body: str) -> None:
    st.html(
        f"""
        <div class="info-card">
            <h3>{escaped(title)}</h3>
            <p>{escaped(body)}</p>
        </div>
        """
    )


def render_chips(items: list[object], tone: str = "") -> None:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        st.caption("未识别到结果：当前内容没有可展示项。")
        return
    class_name = f"chip {tone}".strip()
    chips = "".join(f'<span class="{class_name}">{escaped(item)}</span>' for item in values)
    st.html(f'<div class="chip-row">{chips}</div>')


def render_notice(title: str, body: str, tone: str = "") -> None:
    class_name = f"notice-card {tone}".strip()
    st.html(
        f"""
        <div class="{class_name}">
            <h3>{escaped(title)}</h3>
            <p>{escaped(body)}</p>
        </div>
        """
    )


def render_progress(value: float, tone: str = "") -> None:
    bounded = max(0.0, min(1.0, float(value)))
    percent = round(bounded * 100)
    st.html(
        f"""
        <div class="progress-track">
            <div class="progress-fill {escaped(tone)}" style="width: {percent}%"></div>
        </div>
        """
    )


def render_shell() -> str:
    st.set_page_config(
        page_title="课程评价智能分析平台",
        page_icon=APP_LOGO,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.sidebar.html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title"><span class="sidebar-logo">🎓</span><span>课程评价智能分析平台</span></div>
        </div>
        """
    )
    page = st.sidebar.radio("页面导航", PAGE_OPTIONS, label_visibility="collapsed")
    st.sidebar.html('<div class="sidebar-version">CourseInsight v2.0</div>')
    return page


def page_home() -> None:
    render_hero()

    data_value, data_detail = data_scale_summary()
    model_value, model_detail = active_model_card_value()
    metrics = [
        ("", "数据规模", data_value, data_detail),
        ("", "主模型", model_value, model_detail),
        ("", "支持输入", "单条 / CSV", "支持单条评价分析与批量导入。"),
        ("", "输出结果", "4 类", "情感倾向、关键词、命中维度、教学建议。"),
    ]
    metrics_html = "".join(metric_card_html(*metric) for metric in metrics)
    st.html(f'<div class="home-metric-grid">{metrics_html}</div>')

    render_section_title("从评论文本到教学建议")
    flow_steps = [
        ("01", "输入评价", "支持单条输入和 CSV 批量导入"),
        ("02", "文本清洗", "统一格式，保留中英文关键信息"),
        ("03", "情感识别", "识别积极、中性、消极反馈"),
        ("04", "关键词提取", "提炼课程评价中的高频表达"),
        ("05", "维度识别", "定位教学内容、授课方式、作业任务等问题"),
        ("06", "建议生成", "调用大模型生成可复核的教学改进建议"),
    ]
    flow_html = "".join(
        f"""
        <div class="flow-step">
            <span class="flow-index">{escaped(index)}</span>
            <strong>{escaped(title)}</strong>
            <span>{escaped(detail)}</span>
        </div>
        """
        for index, title, detail in flow_steps
    )
    st.html(f'<div class="flow-grid">{flow_html}</div>')

    render_section_title("系统能力")
    info = [
        ("评价文本分析", "支持单条评价与批量评价的情感识别、关键词提取和维度定位。"),
        ("课程问题定位", "汇总学生高频关注的教学环节，辅助发现课程改进重点。"),
        ("教学改进建议", "结合命中维度与评价证据，生成可复核的教学建议。"),
    ]
    info_html = "".join(
        f"""
        <div class="info-card">
            <h3>{escaped(title)}</h3>
            <p>{escaped(body)}</p>
        </div>
        """
        for title, body in info
    )
    st.html(
        f"""
        <div class="home-info-grid">{info_html}</div>
        <p class="tech-footnote">
            技术栈：Streamlit、pandas、Altair、scikit-learn、multilingual BERT、ChatECNU 兼容接口。
        </p>
        """
    )


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return None


def cell_to_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_dataframe(df: pd.DataFrame) -> list[dict[str, str]]:
    text_col = find_column(list(df.columns), TEXT_COLUMN_CANDIDATES)
    if text_col is None:
        raise ValueError("CSV 必须包含 review_text 字段，或 text/review/comment/content 等兼容文本字段。")

    course_col = find_column(list(df.columns), COURSE_COLUMN_CANDIDATES)
    teacher_col = find_column(list(df.columns), TEACHER_COLUMN_CANDIDATES)
    rating_col = find_column(list(df.columns), RATING_COLUMN_CANDIDATES)
    date_col = find_column(list(df.columns), DATE_COLUMN_CANDIDATES)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, raw in df.iterrows():
        text = cell_to_text(raw.get(text_col, ""))
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(
            {
                "review_text": text,
                "text": text,
                "course_name": cell_to_text(raw.get(course_col, "")) if course_col else "未提供",
                "teacher": cell_to_text(raw.get(teacher_col, "")) if teacher_col else "未提供",
                "rating": cell_to_text(raw.get(rating_col, "")) if rating_col else "",
                "date": cell_to_text(raw.get(date_col, "")) if date_col else "",
            }
        )
    return rows


def normalize_project_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "review_text": text,
                "text": text,
                "course_name": str(row.get("course") or row.get("course_name") or "未提供"),
                "teacher": str(row.get("teacher") or "未提供"),
                "rating": str(row.get("rating") or ""),
                "date": str(row.get("date") or ""),
            }
        )
    return normalized


def load_uploaded_rows(uploaded_file: Any) -> list[dict[str, str]]:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    return normalize_dataframe(df)


def dataset_key(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(row["review_text"] for row in rows)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def collect_dimension_scores(result: dict[str, Any]) -> dict[str, float]:
    for key in ("dimension_scores", "aspect_scores", "topic_scores"):
        values = result.get(key)
        if isinstance(values, dict):
            scores: dict[str, float] = {}
            for aspect, score in values.items():
                try:
                    scores[str(aspect)] = float(score)
                except (TypeError, ValueError):
                    continue
            return scores
    return {}


def advice_source_label(advice: object) -> str:
    if not isinstance(advice, dict):
        return "未启用"
    if advice.get("source") == "llm_api":
        return "大模型生成"
    return "基础建议"


def advice_status_label(advice: object) -> str:
    return "已生成" if isinstance(advice, dict) else "未启用"


def risk_badge_class(risk_level: object) -> str:
    risk = str(risk_level or "").strip()
    if risk == "low":
        return "success"
    if risk == "high":
        return "accent"
    return "warning"


def risk_status_text(risk_level: object) -> str:
    label, _tone = RISK_LABELS.get(str(risk_level or "middle"), RISK_LABELS["middle"])
    if label == "低风险":
        return "保持优势"
    if label == "高风险":
        return "重点问题"
    return "可优化"


def topic_evidence_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in result.get("topic_evidence", []) or []:
        if not isinstance(item, dict):
            continue
        aspect = str(item.get("aspect", "")).strip()
        evidence_text = str(item.get("evidence", "")).strip()
        if aspect and evidence_text and aspect not in seen:
            evidence_items.append(item)
            seen.add(aspect)
    return evidence_items


def keyword_tags_html(values: list[object]) -> str:
    keywords = [str(value).strip() for value in values if str(value).strip()]
    if not keywords:
        return ""
    return "".join(
        f'<span class="detail-keyword">{escaped(keyword)}</span>'
        for keyword in keywords[:5]
    )


def dimension_card_html(item: dict[str, Any]) -> str:
    aspect = str(item.get("aspect", "")).strip() or "课程维度"
    evidence_text = str(item.get("evidence", "")).strip()
    keywords = item.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    chips = keyword_tags_html(keywords)
    keyword_block = f'<div class="detail-keywords">{chips}</div>' if chips else ""
    return f"""
        <div class="detail-card">
            <h3>{escaped(aspect)}</h3>
            {keyword_block}
            <span class="detail-meta">命中依据</span>
            <div class="evidence-quote">“{escaped(evidence_text)}”</div>
        </div>
    """


def advice_suggestions(advice: object) -> list[dict[str, Any]]:
    if not isinstance(advice, dict):
        return []
    suggestions = advice.get("suggestions") or []
    if not isinstance(suggestions, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue
        text = str(suggestion.get("suggestion", "")).strip()
        if text:
            cleaned.append(suggestion)
    return cleaned


def advice_card_html(suggestion: dict[str, Any], risk_level: object) -> str:
    aspect = str(suggestion.get("aspect", "")).strip() or "教学改进"
    evidence = str(suggestion.get("evidence", "")).strip() or "根据本条评价内容生成。"
    badge = risk_status_text(risk_level)
    badge_class = risk_badge_class(risk_level)
    return f"""
        <div class="advice-card">
            <div class="badge-row"><span class="ui-badge {escaped(badge_class)}">{escaped(badge)}</span></div>
            <h3>{escaped(aspect)}</h3>
            <p>{escaped(suggestion.get("suggestion", ""))}</p>
            <span class="advice-meta">依据</span>
            <div class="evidence-quote">“{escaped(evidence)}”</div>
        </div>
    """


def render_backend_error(error: dict[str, str]) -> None:
    st.html(
        """
        <div class="error-card">
            <h3>分析失败</h3>
            <p>分析失败，请检查模型文件或大模型配置后重试。</p>
        </div>
        """
    )
    with st.expander("查看技术详情"):
        st.write(error.get("message", "分析过程没有返回更多信息"))


def render_single_result(result: dict[str, Any]) -> None:
    sentiment = str(result.get("sentiment", "neutral"))
    confidence = float(result.get("confidence", 0) or 0)
    advice = result.get("llm_advice")
    keywords = [str(item).strip() for item in result.get("keywords", []) if str(item).strip()]
    keyword_detail = "、".join(keywords[:4]) if keywords else "本条评价未提取到明显关键词。"

    render_section_title("分析结果总览")
    overview = [
        ("", "情感倾向", SENTIMENT_LABELS.get(sentiment, sentiment), "根据评价文本判断整体反馈倾向。"),
        ("", "置信度", format_percent(confidence), "模型对当前判断的置信程度。"),
        ("", "关键词", count_with_unit(len(keywords), "个"), keyword_detail),
        ("", "建议状态", advice_status_label(advice), "已结合评价内容生成教学建议。" if isinstance(advice, dict) else "本次未启用建议生成。"),
    ]
    overview_html = "".join(metric_card_html(*metric) for metric in overview)
    st.html(f'<div class="single-overview-grid">{overview_html}</div>')

    render_section_title("命中维度与证据")
    evidence_items = topic_evidence_items(result)
    if evidence_items:
        dimension_html = "".join(dimension_card_html(item) for item in evidence_items)
        st.html(f'<div class="detail-card-grid">{dimension_html}</div>')
    else:
        render_notice("未识别到明确维度", "本条评价较短或表达较泛，可补充具体课程环节后重新分析。")

    render_section_title("教学改进建议")
    if isinstance(advice, dict):
        risk_label = risk_status_text(advice.get("risk_level"))
        risk_class = risk_badge_class(advice.get("risk_level"))
        st.html(
            f"""
            <div class="analysis-summary-card advice-highlight">
                <div class="badge-row">
                    <span class="ui-badge accent">{escaped(advice_source_label(advice))}</span>
                </div>
                <p>{escaped(advice.get("summary", "本次建议未生成总结。"))}</p>
            </div>
            """
        )
        suggestions = advice_suggestions(advice)
        if suggestions:
            advice_html = "".join(
                advice_card_html(suggestion, advice.get("risk_level"))
                for suggestion in suggestions[:3]
            )
            st.html(f'<div class="advice-card-grid">{advice_html}</div>')
    else:
        st.html(
            """
            <div class="analysis-summary-card advice-highlight">
                <p>本次未启用建议生成。开启“生成改进建议”后，系统将结合评价内容给出可复核的教学建议。</p>
            </div>
            """
        )


def page_single_analysis() -> None:
    render_page_header(
        "📝",
        "单条分析",
        "输入一条课程评价，查看模型给出的情感判断、关键词、维度证据和改进建议。",
    )

    if "single_review_text" not in st.session_state:
        st.session_state["single_review_text"] = SAMPLE_REVIEWS["中文评价"]

    with st.container(border=True):
        render_card_title("输入课程评价", "可直接输入，也可以选择下方示例评价。")
        example_cols = st.columns(3, gap="small")
        for index, (label, sample_text) in enumerate(SAMPLE_REVIEWS.items()):
            col = example_cols[index]
            with col:
                if st.button(label, key=f"single_example_{index}", width="stretch"):
                    st.session_state["single_review_text"] = sample_text
                    st.session_state.pop("single_result", None)
                    st.session_state.pop("single_result_text", None)
                    st.session_state.pop("single_error", None)
                    st.session_state.pop("single_error_text", None)
                    st.rerun()

        with st.form("single_analysis_form", border=False):
            text = st.text_area(
                "评价文本",
                key="single_review_text",
                height=160,
                placeholder="请输入学生对课程内容、教学方法、课堂互动、作业反馈或学习收获的评价。",
            )
            use_llm = st.toggle("生成改进建议", value=True, help="启用后将结合评价内容生成教学改进建议。")
            submitted = st.form_submit_button("开始分析", type="primary", width="stretch")

    if submitted:
        if not text.strip():
            st.warning("请输入评价文本后再开始分析。")
        else:
            try:
                with st.spinner("正在分析评价文本，请稍候..."):
                    st.session_state["single_result"] = run_single_analysis(text.strip(), use_llm=use_llm)
                    st.session_state["single_result_text"] = text.strip()
                    st.session_state.pop("single_error", None)
                    st.session_state.pop("single_error_text", None)
            except Exception as exc:
                st.session_state.pop("single_result", None)
                st.session_state.pop("single_result_text", None)
                st.session_state["single_error"] = single_backend_error_payload(exc)
                st.session_state["single_error_text"] = text.strip()
            st.rerun()

    current_text = str(st.session_state.get("single_review_text", "")).strip()
    error = st.session_state.get("single_error")
    error_text = str(st.session_state.get("single_error_text", "")).strip()
    result = st.session_state.get("single_result")
    result_text = str(st.session_state.get("single_result_text", "")).strip()
    if error and error_text == current_text:
        render_backend_error(error)
    elif result and result_text == current_text:
        render_single_result(result)
    else:
        render_notice("请先输入评价文本", "输入课程评价并点击“开始分析”后，系统将展示情感倾向、关键词、命中维度和改进建议。")


def sentiment_score(sentiment: object) -> float:
    return {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(str(sentiment), 0.5)


def sentiment_chart(sentiments: dict[str, int]) -> alt.Chart:
    frame = pd.DataFrame(
        [
            {"情感": SENTIMENT_LABELS.get(key, key), "数量": value, "key": key}
            for key, value in sentiments.items()
        ]
    )
    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=55, outerRadius=92)
        .encode(
            theta=alt.Theta("数量:Q"),
            color=alt.Color(
                "情感:N",
                scale=alt.Scale(
                    domain=["积极", "中性", "消极"],
                    range=[
                        CHART_COLORS["success"],
                        CHART_COLORS["warning"],
                        CHART_COLORS["danger"],
                    ],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["情感:N", "数量:Q"],
        )
        .properties(height=260)
    )


def bar_chart(frame: pd.DataFrame, x_col: str, y_col: str, color: str = CHART_COLORS["primary"]) -> alt.Chart:
    if frame.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, color=color)
        .encode(
            y=alt.Y(f"{x_col}:N", sort="-x", title=None),
            x=alt.X(f"{y_col}:Q", title=None),
            tooltip=[alt.Tooltip(f"{x_col}:N"), alt.Tooltip(f"{y_col}:Q")],
        )
        .properties(height=max(220, 30 * len(frame)))
        .configure_axis(labelColor=CHART_COLORS["muted"], titleColor=CHART_COLORS["muted"])
        .configure_view(strokeWidth=0)
    )


def result_rows(rows: list[dict[str, str]], results: list[dict[str, Any]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row, result in zip(rows, results):
        output.append(
            {
                "评价文本": row.get("review_text", ""),
                "课程": row.get("course_name", "未提供") or "未提供",
                "教师": row.get("teacher", "未提供") or "未提供",
                "评分": row.get("rating", ""),
                "日期": row.get("date", ""),
                "情感": SENTIMENT_LABELS.get(str(result.get("sentiment")), str(result.get("sentiment"))),
                "置信度": float(result.get("confidence", 0) or 0),
                "课程维度": "、".join(str(item) for item in result.get("topics", [])),
                "关键词": "、".join(str(item) for item in result.get("keywords", [])),
                "分析方法": SOURCE_LABELS.get(str(result.get("sentiment_source", "")), str(result.get("sentiment_source", ""))),
            }
        )
    return output


def rating_summary_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or "rating" not in frame:
        return pd.DataFrame()
    frame["rating_num"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["rating_num"])
    if frame.empty:
        return pd.DataFrame()
    group_col = "course_name" if frame["course_name"].replace("", pd.NA).notna().any() else "teacher"
    summary = (
        frame.groupby(group_col, dropna=False)
        .agg(平均评分=("rating_num", "mean"), 评论数=("review_text", "count"))
        .reset_index()
        .rename(columns={group_col: "对象"})
        .sort_values(["平均评分", "评论数"], ascending=[True, False])
    )
    return summary


def result_indexes_by_sentiment(results: list[dict[str, Any]], labels: set[str]) -> list[int]:
    return [
        index
        for index, result in enumerate(results)
        if str(result.get("sentiment", "neutral")) in labels
    ]


def subset_by_indexes(items: list[Any], indexes: list[int]) -> list[Any]:
    return [items[index] for index in indexes if 0 <= index < len(items)]


def batch_summary_text(
    sentiments: dict[str, int],
    topics: dict[str, int],
    problem_topics: dict[str, int],
) -> str:
    if not sentiments or sum(sentiments.values()) == 0:
        return "当前批次尚未形成可展示的统计结论。"

    dominant = max(sentiments.items(), key=lambda item: item[1])[0]
    if dominant == "positive":
        summary = "本批评价整体偏积极。"
    elif dominant == "negative":
        summary = "本批评价中待关注问题较多，建议优先查看典型问题评论。"
    else:
        summary = "本批评价整体反馈较为中性，包含认可与改进意见。"

    if topics:
        top_topic = next(iter(topics))
        summary += f" 学生主要关注 {top_topic}。"
    else:
        summary += " 当前未识别到明确高频维度。"

    if problem_topics:
        problem_topic = next(iter(problem_topics))
        summary += f" 其中，{problem_topic} 是当前更需要关注的改进方向。"
    else:
        summary += " 当前未识别出明显高关注问题维度。"
    return summary


def render_batch_summary(
    sentiments: dict[str, int],
    topics: dict[str, int],
    problem_topics: dict[str, int],
) -> None:
    st.html(
        f"""
        <div class="analysis-summary-card">
            <div class="badge-row"><span class="ui-badge primary">本批结论摘要</span></div>
            <p>{escaped(batch_summary_text(sentiments, topics, problem_topics))}</p>
        </div>
        """
    )


def result_display_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["评价文本", "情感", "置信度", "课程维度", "关键词", "课程"]
    optional = ["教师", "评分", "日期"]
    columns = [column for column in preferred if column in frame.columns]
    for column in optional:
        if column in frame.columns and frame[column].replace("", pd.NA).dropna().any():
            columns.append(column)
    return columns


def render_batch_results(rows: list[dict[str, str]], results: list[dict[str, Any]], source_name: str) -> None:
    sentiments = sentiment_distribution(results)
    topics = topic_distribution(results)
    total = len(results)
    positive_rate = sentiments.get("positive", 0) / total if total else 0
    attention_indexes = result_indexes_by_sentiment(results, {"negative"})
    attention_results = subset_by_indexes(results, attention_indexes)
    attention_rows = subset_by_indexes(rows, attention_indexes)
    problem_topics = topic_distribution(attention_results)
    keyword_indexes = attention_indexes
    if len(keyword_indexes) < 3:
        keyword_indexes = result_indexes_by_sentiment(results, {"negative", "neutral"})
    if not keyword_indexes:
        keyword_indexes = list(range(len(results)))
    keyword_rows = subset_by_indexes(rows, keyword_indexes)
    keyword_texts = [row["review_text"] for row in keyword_rows]
    keywords = dict(extract_keywords(keyword_texts, top_k=12))
    top_topic = next(iter(topics), "未识别到结果")

    render_section_title("批量概览")
    metric_cols = st.columns(4)
    batch_metrics = [
        ("", "评论总数", count_with_unit(total, "条"), f"数据来源：{source_name}"),
        ("", "积极反馈占比", format_percent(positive_rate), "反映学生整体认可程度。"),
        ("", "待关注评论", count_with_unit(len(attention_results), "条"), "当前识别为消极的评价数量。"),
        ("", "主要关注维度", top_topic, "本批评价中被提及最多的教学维度。"),
    ]
    for col, metric in zip(metric_cols, batch_metrics):
        with col:
            render_metric_card(*metric)

    render_batch_summary(sentiments, topics, problem_topics)

    render_section_title("整体分布")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_card_title("情感分布")
            st.altair_chart(sentiment_chart(sentiments), width="stretch")
    with right:
        with st.container(border=True):
            render_card_title("维度关注排行", "统计全部评价中被识别出的高频维度。")
            topic_frame = pd.DataFrame(
                [{"维度": key, "次数": value} for key, value in list(topics.items())[:8]]
            )
            if topic_frame.empty:
                st.info("未识别到结果：当前内容未识别到明确维度。")
            else:
                st.altair_chart(bar_chart(topic_frame, "维度", "次数"), width="stretch")

    render_section_title("问题定位")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            render_card_title("高频问题关键词", "从待关注评论中提取高频表达，用于辅助定位具体问题。")
            keyword_frame = pd.DataFrame(
                [{"关键词": key, "次数": value} for key, value in keywords.items()]
            )
            if keyword_frame.empty:
                st.info("未识别到结果：当前内容未识别到明确关键词。")
            else:
                st.altair_chart(
                    bar_chart(keyword_frame, "关键词", "次数", color=CHART_COLORS["accent"]),
                    width="stretch",
                )
    with chart_right:
        with st.container(border=True):
            render_card_title("高频问题维度")
            topic_frame = pd.DataFrame(
                [{"维度": key, "出现次数": value} for key, value in list(problem_topics.items())[:8]]
            )
            if topic_frame.empty:
                st.info("未识别到结果：当前批次未识别出需要重点关注的问题维度。")
            else:
                st.altair_chart(
                    bar_chart(topic_frame, "维度", "出现次数", color=CHART_COLORS["accent"]),
                    width="stretch",
                )

    render_section_title("典型评论证据")
    table_left, table_right = st.columns(2)
    output_rows = result_rows(rows, results)
    result_frame = pd.DataFrame(output_rows)
    with table_left:
        with st.container(border=True):
            render_card_title("典型认可评论")
            positive_df = result_frame[result_frame["情感"] == "积极"].sort_values("置信度", ascending=False).head(5)
            if positive_df.empty:
                st.info("未识别到结果：当前批次未识别出高置信度认可评论。")
            else:
                positive_columns = [column for column in ["评价文本", "课程", "课程维度", "关键词"] if column in positive_df.columns]
                st.dataframe(
                    positive_df[positive_columns],
                    hide_index=True,
                    width="stretch",
                    column_config={"评价文本": st.column_config.TextColumn("评价文本", width="large")},
                )
    with table_right:
        with st.container(border=True):
            render_card_title("典型问题评论")
            negative_df = result_frame[result_frame["情感"] == "消极"].sort_values("置信度", ascending=False).head(5)
            if negative_df.empty:
                st.info("未识别到结果：当前批次未识别出高置信度问题评论。")
            else:
                st.dataframe(
                    negative_df[["评价文本", "课程", "置信度", "课程维度"]],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "评价文本": st.column_config.TextColumn("评价文本", width="large"),
                        "置信度": st.column_config.ProgressColumn("置信度", min_value=0, max_value=1, format="%.2f"),
                    },
                )

    with st.container(border=True):
        render_card_title("批量分析明细")
        display_columns = result_display_columns(result_frame)
        st.dataframe(
            result_frame[display_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "评价文本": st.column_config.TextColumn("评价文本", width="large"),
                "置信度": st.column_config.ProgressColumn("置信度", min_value=0, max_value=1, format="%.2f"),
            },
        )
        st.download_button(
            "下载分析结果 CSV",
            data=result_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="courseinsight_batch_results.csv",
            mime="text/csv",
            width="stretch",
        )


def page_batch_analysis() -> None:
    render_page_header(
        "📊",
        "批量分析",
        "上传课程评价 CSV 或加载示例数据，生成情感分布、维度统计、关键词和分析明细。",
    )

    uploaded_file = None
    rows: list[dict[str, str]] = []
    source_name = ""
    with st.container(border=True):
        render_card_title("上传区", "CSV 需包含评价文本字段，支持 review_text、text、review、comment、content 等常见列名。")
        uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
        load_sample = st.button("加载示例数据", width="stretch")
        if load_sample:
            st.session_state["batch_sample_rows"] = normalize_project_rows(load_sample_rows())
            st.session_state["batch_sample_source"] = "项目示例数据"
            st.rerun()

    if uploaded_file is not None:
        try:
            rows = load_uploaded_rows(uploaded_file)
            source_name = uploaded_file.name
        except Exception as exc:
            st.error(f"CSV 读取失败：{exc}")
            return
    else:
        rows = st.session_state.get("batch_sample_rows", [])
        source_name = st.session_state.get("batch_sample_source", "")

    if not rows:
        render_notice("等待数据", "请上传 CSV，或点击“加载项目已有示例数据”。")
        return

    preview = pd.DataFrame(rows)
    with st.container(border=True):
        render_card_title("数据预览", "开始分析前可检查文本字段和课程信息。")
        st.dataframe(preview.head(10), hide_index=True, width="stretch")

    key = dataset_key(rows)
    has_current_result = (
        st.session_state.get("batch_dataset_key") == key
        and bool(st.session_state.get("batch_results"))
    )
    if st.button("重新分析当前数据" if has_current_result else "开始批量分析", type="primary", width="stretch"):
        texts = tuple(row["review_text"] for row in rows)
        with st.spinner("正在分析批量评价，请稍候..."):
            st.session_state["batch_results"] = run_batch_analysis_cached(texts, current_model_version())
            st.session_state["batch_dataset_key"] = key
            st.session_state["batch_rows"] = rows
            st.session_state["batch_source_name"] = source_name
        st.rerun()

    if not has_current_result:
        render_notice("数据已就绪", "点击“开始批量分析”后会生成情感分布、维度统计、关键词和分析明细。")
        return

    results = st.session_state.get("batch_results", [])
    stored_rows = st.session_state.get("batch_rows", rows)
    stored_source = st.session_state.get("batch_source_name", source_name)
    render_batch_results(stored_rows, results, stored_source)


def macro_report(metrics: dict[str, Any]) -> dict[str, Any]:
    report = metrics.get("classification_report", {})
    macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
    return macro if isinstance(macro, dict) else {}


def macro_f1_value(metrics: dict[str, Any]) -> object:
    macro = macro_report(metrics)
    return macro.get("f1-score") or metrics.get("macro_f1")


def classification_support_total(metrics: dict[str, Any]) -> int:
    report = metrics.get("classification_report", {})
    if not isinstance(report, dict):
        return 0
    total = 0
    for label, values in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"} or not isinstance(values, dict):
            continue
        try:
            total += int(values.get("support", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def current_eval_metrics() -> tuple[str, dict[str, Any]]:
    bert_metrics = load_json_file(str(BERT_METRICS))
    if bert_metrics:
        return "BERT", bert_metrics
    return "传统模型", load_json_file(str(MODEL_METRICS))


def model_comparison_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bert_metrics = load_json_file(str(BERT_METRICS))
    if bert_metrics:
        rows.append(
            {
                "模型": "BERT",
                "Accuracy": bert_metrics.get("accuracy"),
                "Macro-F1": macro_f1_value(bert_metrics),
                "模型类型": "当前主模型",
            }
        )
    model_metrics = load_json_file(str(MODEL_METRICS))
    for name, values in model_metrics.get("results", {}).items():
        report = values if isinstance(values, dict) else {}
        rows.append(
            {
                "模型": display_model_name(name),
                "Accuracy": report.get("accuracy"),
                "Macro-F1": macro_f1_value(report),
                "模型类型": "对比模型",
            }
        )
    return pd.DataFrame(rows)


def classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    if not isinstance(report, dict):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for label, values in report.items():
        if label == "accuracy" or not isinstance(values, dict):
            continue
        rows.append(
            {
                "类别": SENTIMENT_LABELS.get(label, label),
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1-score": values.get("f1-score"),
                "Support": values.get("support"),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    matrix = metrics.get("confusion_matrix")
    labels = metrics.get("labels", [])
    if not matrix or not labels:
        return pd.DataFrame()
    readable_labels = [SENTIMENT_LABELS.get(str(label), str(label)) for label in labels]
    return pd.DataFrame(matrix, index=readable_labels, columns=readable_labels)


def model_metric_chart(frame: pd.DataFrame) -> alt.Chart:
    if frame.empty:
        return alt.Chart(pd.DataFrame(columns=["模型", "指标", "分数"])).mark_bar()
    long_frame = frame.melt(
        id_vars=["模型"],
        value_vars=["Accuracy", "Macro-F1"],
        var_name="指标",
        value_name="分数",
    ).dropna()
    return (
        alt.Chart(long_frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("模型:N", sort=list(frame["模型"]), title=None),
            x=alt.X("分数:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "指标:N",
                scale=alt.Scale(
                    domain=["Accuracy", "Macro-F1"],
                    range=[CHART_COLORS["primary"], CHART_COLORS["compare"]],
                ),
                legend=alt.Legend(title=None),
            ),
            yOffset=alt.YOffset("指标:N"),
            tooltip=["模型:N", "指标:N", alt.Tooltip("分数:Q", format=".4f")],
        )
        .properties(height=max(220, 42 * frame["模型"].nunique()))
        .configure_axis(labelColor=CHART_COLORS["muted"], titleColor=CHART_COLORS["muted"])
        .configure_view(strokeWidth=0)
    )


def page_model_eval() -> None:
    render_page_header(
        "🧠",
        "模型评估",
        "本页展示已完成实验的评估结果，用于说明模型有效性。",
    )

    model_name, metrics = current_eval_metrics()
    metric_cols = st.columns(4)
    support_total = classification_support_total(metrics)
    cards = [
        ("", "当前主模型", model_name, "用于课程评价情感分类。"),
        ("", "测试集准确率", format_percent(metrics.get("accuracy")), "模型整体分类正确比例。"),
        ("", "Macro-F1", format_decimal(macro_f1_value(metrics)), "综合衡量三类情感分类效果。"),
        ("", "评估样本", count_with_unit(support_total, "条") if support_total else "缺少数据", "用于当前分类报告的测试样本数。"),
    ]
    for col, card in zip(metric_cols, cards):
        with col:
            render_metric_card(*card)

    render_section_title("模型对比表")
    comparison = model_comparison_frame()
    with st.container(border=True):
        render_card_title("模型指标对比", "当前主模型为 BERT；传统机器学习模型作为对照。")
        if comparison.empty:
            st.info("缺少评估文件：当前未找到模型对比指标。")
        else:
            st.altair_chart(model_metric_chart(comparison), width="stretch")
            st.dataframe(
                comparison,
                hide_index=True,
                width="stretch",
                column_config={
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                    "Macro-F1": st.column_config.NumberColumn("Macro-F1", format="%.4f"),
                },
            )

    render_section_title("混淆矩阵与分类报告")
    left, right = st.columns(2)
    model_metrics = load_json_file(str(MODEL_METRICS))
    with left:
        with st.container(border=True):
            render_card_title("混淆矩阵", "用于观察不同情感类别之间的识别情况。")
            current_matrix = confusion_matrix_frame(metrics)
            if model_name == "BERT" and not current_matrix.empty:
                st.caption("BERT 混淆矩阵")
                st.dataframe(current_matrix, width="stretch")
            elif model_name == "BERT" and BERT_CONFUSION_MATRIX_CHART.exists():
                st.image(str(BERT_CONFUSION_MATRIX_CHART), caption="BERT 混淆矩阵", width="stretch")
            elif CONFUSION_MATRIX_CHART.exists():
                st.image(str(CONFUSION_MATRIX_CHART), caption="模型评估混淆矩阵", width="stretch")
            else:
                fallback_matrix = confusion_matrix_frame(model_metrics)
                if model_name == "BERT" or fallback_matrix.empty:
                    st.info("缺少评估文件：当前未找到 BERT 混淆矩阵。")
                else:
                    st.caption("传统模型混淆矩阵（对比）")
                    st.dataframe(fallback_matrix, width="stretch")
    with right:
        with st.container(border=True):
            render_card_title("分类报告", "展示不同情感类别下的识别表现。")
            report_frame = classification_report_frame(metrics)
            if report_frame.empty:
                st.info("缺少评估文件：当前未找到可展示的分类报告。")
            else:
                st.dataframe(
                    report_frame,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                        "Recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                        "F1-score": st.column_config.NumberColumn("F1-score", format="%.4f"),
                        "Support": st.column_config.NumberColumn("Support", format="%.0f"),
                    },
                )

    render_section_title("模型选择说明")
    cols = st.columns(3)
    explanations = [
        ("为什么选择 BERT", "BERT 能更好处理中英文混合评价和上下文表达，适合作为课程评价情感分类主模型。"),
        ("模型优势", "结合课程场景规则校正，可增强否定、转折和关键教学场景的判断稳定性。"),
        ("后续优化", "对于讽刺表达、隐含抱怨和更细粒度教学问题，还需要更多标注数据支持。"),
    ]
    for col, item in zip(cols, explanations):
        with col:
            render_info_card(*item)

    ablation_metrics = load_json_file(str(ABLATION_METRICS))
    if ablation_metrics.get("experiments"):
        with st.container(border=True):
            render_card_title("补充实验：组件贡献分析", "用于展示不同组件对系统表现的影响。")
            rows = [
                {
                    "实验版本": name,
                    "Accuracy": values.get("accuracy"),
                    "Macro-F1": values.get("macro_f1"),
                    "通过数": values.get("passed"),
                    "失败数": values.get("failed"),
                    "状态": values.get("status"),
                }
                for name, values in ablation_metrics.get("experiments", {}).items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def main() -> None:
    page = render_shell()
    if page == "🏠 首页概览":
        page_home()
    elif page == "📄 单条分析":
        page_single_analysis()
    elif page == "📊 批量分析":
        page_batch_analysis()
    else:
        page_model_eval()


if __name__ == "__main__":
    main()
