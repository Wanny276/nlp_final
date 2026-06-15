"""CourseInsight Streamlit front end."""

from __future__ import annotations

import hashlib
import html
import json
import re
from io import BytesIO
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


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = Path("data/sample_reviews.csv")
TEST_CASES = Path("data/test_cases.csv")
MODEL_METRICS = Path("models/model_metrics.json")
BERT_METRICS = Path("outputs/bert_metrics_final.json")
ABLATION_METRICS = Path("outputs/reports/final_model/ablation_metrics.json")
APP_LOGO = "🎓"
BRAND_NAME = "CourseInsight"
BRAND_SUBTITLE = "面向课程评论的中英双语智能分析系统"

PAGE_OPTIONS = [
    "🏠 首页概览",
    "📄 单条分析",
    "📊 批量分析",
    "✅ 固定案例验证",
    "🧠 模型评估",
]

SAMPLE_REVIEWS = {
    "中文评价": "老师讲课很清楚，课程内容和知识点组织得很好，案例也帮助我理解重点，整体收获很多。",
    "英文评价": "The assignments are too many, the deadlines are stressful, and the exam scope is not clear enough.",
    "中英混合": "讲得很清楚，但 final project 有点赶，debug 花了不少时间。",
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
DEFAULT_SENTIMENT_LABEL_ORDER = ("negative", "neutral", "positive")
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
    "primary": "#5636D3",
    "accent": "#6D4AEF",
    "success": "#28C76F",
    "warning": "#E67835",
    "danger": "#D95C59",
    "muted": "#596076",
    "compare": "#B6AFE4",
    "accent_2": "#1660BC",
    "bert_blue": "#6EA2D8",
    "bert_blue_soft": "#BBD0EA",
    "bert_blue_dark": "#174F8A",
    "bert_blue_pale": "#F4F8FC",
}

APP_CSS = """
<style>
:root {
    --primary: #5636D3;
    --primary-2: #3E20B5;
    --accent: #6D4AEF;
    --accent-2: #1660BC;
    --primary-soft: #F3F0FF;
    --accent-soft: #F5F2FF;
    --background: #F8F7FC;
    --card-bg: #FFFFFF;
    --text-main: #080C35;
    --text-muted: #596076;
    --border: #E4DDF6;
    --success: #28C76F;
    --warning: #E67835;
    --danger: #D95C59;
    --sidebar-bg: #080C35;
    --card-shadow: 0 10px 24px rgba(8, 12, 53, 0.08);
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
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, #120A45 100%);
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
    padding: 0.35rem 0.95rem 1.05rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.sidebar-brand-title {
    display: block;
    margin: 0;
    color: #FFFFFF;
    font-size: 21px;
    font-weight: 800;
    line-height: 1.25;
    letter-spacing: 0.01em;
}

.sidebar-brand-subtitle {
    display: block;
    margin-top: 0.28rem;
    color: rgba(255, 255, 255, 0.78);
    font-size: 15px;
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

.brand-block {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    min-width: 0;
}

.brand-text {
    min-width: 0;
}

.brand-title {
    margin: 0;
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: 0.01em;
}

.brand-subtitle {
    margin-top: 0.25rem;
    line-height: 1.35;
    font-weight: 600;
}

.brand-sidebar {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.55rem;
    text-align: left;
}

.brand-sidebar .brand-text {
    width: 100%;
    text-align: left;
}

.brand-sidebar .brand-title {
    color: #FFFFFF;
    font-size: 20px;
}

.brand-sidebar .brand-subtitle {
    max-width: none;
    margin-left: 0;
    margin-right: 0;
    color: rgba(255, 255, 255, 0.76);
    font-size: 11px;
    letter-spacing: -0.01em;
    white-space: nowrap;
}

.brand-hero {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.7rem;
    text-align: left;
}

.brand-hero .brand-text {
    width: 100%;
    text-align: left;
}

.brand-hero .brand-title {
    color: var(--text-main);
    font-size: 32px;
}

.brand-hero .brand-subtitle {
    color: var(--text-muted);
    font-size: 15px;
}

.ci-logo {
    --logo-size: 48px;
    position: relative;
    flex: 0 0 var(--logo-size);
    width: var(--logo-size);
    height: var(--logo-size);
}

.brand-sidebar .ci-logo {
    --logo-size: 46px;
}

.brand-hero .ci-logo {
    --logo-size: 58px;
}

.ci-logo-box {
    position: absolute;
    inset: 0;
    overflow: hidden;
    border: 1px solid rgba(205, 189, 255, 0.42);
    border-radius: 24%;
    background: rgba(255, 255, 255, 0.10);
    box-shadow: none;
    box-sizing: border-box;
}

.brand-sidebar .ci-logo-box {
    border-color: rgba(232, 225, 255, 0.58);
    background: linear-gradient(135deg, #F8F5FF 0%, #E8E1FF 100%);
    box-shadow: 0 10px 24px rgba(5, 8, 35, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.65);
}

.brand-hero .ci-logo-box {
    border-color: rgba(86, 54, 211, 0.18);
    background: linear-gradient(135deg, #FFFFFF 0%, #F5F2FF 100%);
    box-shadow: 0 8px 18px rgba(86, 54, 211, 0.08);
}

.ci-logo-box::after,
.ci-logo-box::before {
    display: none;
}

.ci-monogram {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: calc(var(--logo-size) * 0.015);
    padding-right: calc(var(--logo-size) * 0.02);
    box-sizing: border-box;
    font-family: "Arial Black", "Arial", "Microsoft YaHei", sans-serif;
    font-size: calc(var(--logo-size) * 0.48);
    font-weight: 900;
    line-height: 1;
    letter-spacing: calc(var(--logo-size) * -0.055);
    transform: translateY(-1%);
    user-select: none;
}

.ci-monogram span {
    display: inline-block;
    color: #5636D3 !important;
}

.ci-monogram-c {
    color: inherit !important;
}

.ci-monogram-i {
    color: inherit !important;
    margin-left: calc(var(--logo-size) * 0.02);
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

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
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

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
    width: 100%;
    margin: 0 !important;
    color: rgba(255, 255, 255, 0.78);
    font-size: 14px;
    line-height: 1.3;
    font-weight: 600;
    text-align: left;
    white-space: nowrap;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
    border-color: rgba(255, 255, 255, 0.18);
    box-shadow: 0 10px 24px rgba(8, 12, 53, 0.26);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
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
    margin: 0.8rem 0 0;
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

.runtime-status-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    align-items: stretch;
}

.runtime-status-grid .metric-card {
    min-height: 150px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.runtime-status-grid .metric-value {
    font-size: 32px;
    white-space: normal;
    overflow-wrap: anywhere;
}

.runtime-status-grid .metric-detail {
    overflow-wrap: anywhere;
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

.analysis-summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    align-items: stretch;
}

.detail-card-grid,
.advice-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
    position: relative;
    padding-top: 52px;
}

.advice-card {
    min-height: 202px;
}

.advice-card > .badge-row {
    position: absolute;
    top: 28px;
    right: 28px;
    margin: 0;
}

.advice-card > .badge-row .ui-badge {
    font-size: 12px;
    min-height: 20px;
    padding: 0.08rem 0.42rem;
}

.advice-summary-card {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 18px;
    padding: 18px 20px;
    border: 1px solid rgba(86, 54, 211, 0.18);
    border-left: 4px solid var(--primary);
    border-radius: var(--radius-card);
    background: linear-gradient(135deg, #FFFFFF 0%, #F6F3FF 100%);
    box-shadow: 0 8px 20px rgba(8, 12, 53, 0.06);
}

.advice-summary-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 32px;
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: var(--primary-soft);
    color: var(--primary);
    font-size: 16px;
    font-weight: 800;
}

.advice-summary-content {
    min-width: 0;
}

.advice-summary-meta {
    margin-bottom: 6px;
    color: var(--primary);
    font-size: 13px;
    font-weight: 800;
    line-height: 1.35;
}

.advice-summary-card p {
    margin: 0;
    color: var(--text-main);
    font-size: 15px;
    line-height: 1.75;
    font-weight: 500;
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
    min-height: 20px;
    padding: 0.08rem 0.42rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--card-bg);
    color: var(--text-muted);
    font-size: 12px;
    line-height: 1.25;
    font-weight: 600;
}

.ui-badge.primary {
    border-color: rgba(86, 54, 211, 0.22);
    background: var(--primary-soft);
    color: var(--primary);
}

.ui-badge.accent {
    border-color: rgba(109, 74, 239, 0.25);
    background: var(--accent-soft);
    color: var(--accent);
}

.ui-badge.warning {
    border-color: rgba(230, 120, 53, 0.26);
    background: #FFF8F1;
    color: #B85B20;
}

.ui-badge.success {
    border-color: rgba(40, 199, 111, 0.24);
    background: #F0FFF6;
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
    background: rgba(86, 54, 211, 0.30);
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

.notice-card.success {
    border-left: 4px solid var(--success) !important;
}

.notice-card.danger {
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
    box-shadow: 0 0 0 3px rgba(86, 54, 211, 0.14) !important;
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

[data-testid="stTable"] {
    overflow: hidden;
    border: 1px solid #D9E8F8;
    border-radius: var(--radius-control);
}

@media (max-width: 1000px) {
    .home-metric-grid,
    .runtime-status-grid,
    .home-info-grid,
    .single-overview-grid,
    .analysis-summary-grid {
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
    .runtime-status-grid,
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

    .brand-hero .ci-logo {
        --logo-size: 50px;
    }

    .brand-hero .brand-title {
        font-size: 26px;
    }

    .brand-hero .brand-subtitle {
        font-size: 14px;
    }

    .hero-copy {
        margin-left: 0;
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


def courseinsight_logo_html() -> str:
    return """
    <div class="ci-logo" aria-hidden="true">
        <div class="ci-logo-box"></div>
        <div class="ci-monogram"><span class="ci-monogram-c">C</span><span class="ci-monogram-i">I</span></div>
    </div>
    """

def courseinsight_brand_html(variant: str) -> str:
    return f'''
    <div class="brand-block brand-{escaped(variant)}">
        {courseinsight_logo_html()}
        <div class="brand-text">
            <div class="brand-title">{escaped(BRAND_NAME)}</div>
            <div class="brand-subtitle">{escaped(BRAND_SUBTITLE)}</div>
        </div>
    </div>
    '''


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
def load_test_case_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


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


def backend_display_name(name: object) -> str:
    normalized = str(name or "").strip().lower()
    labels = {
        "auto": "AUTO",
        "bert": "BERT",
        "tfidf": "TF-IDF",
        "rule": "规则",
    }
    return labels.get(normalized, str(name or "未知"))


def bert_runtime_status_label(bert: dict[str, Any]) -> str:
    if not bert.get("dependencies_available"):
        return "依赖不可用"
    if bert.get("error"):
        return "不可用"
    if bert.get("ready"):
        return "已就绪"
    if not bert.get("model_available"):
        return "权重未就绪"
    return "待加载"


def project_relative_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "未配置路径"
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return text


def bert_runtime_status_detail(bert: dict[str, Any]) -> str:
    error = str(bert.get("error") or "").strip()
    if error:
        return error
    return project_relative_path(bert.get("model_path"))


def tfidf_runtime_status_label(available: object) -> str:
    return "已就绪" if available else "不可用"


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
                {courseinsight_brand_html("hero")}
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


def render_metric_grid(cards: list[tuple[str, str, object, str]], class_name: str = "runtime-status-grid") -> None:
    cards_html = "".join(metric_card_html(*card) for card in cards)
    st.html(f'<div class="{class_name}">{cards_html}</div>')


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
        page_title="面向课程评论的中英双语智能分析系统",
        page_icon=APP_LOGO,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.sidebar.html(
        f"""
        <div class="sidebar-brand">
            {courseinsight_brand_html("sidebar")}
        </div>
        """
    )
    page = st.sidebar.radio(" ", PAGE_OPTIONS, label_visibility="collapsed")
    st.sidebar.html('<div class="sidebar-version">CourseInsight v2.0</div>')
    return page


def page_home() -> None:
    render_hero()

    data_value, data_detail = data_scale_summary()
    model_value, model_detail = active_model_card_value()
    metrics = [
        ("", "数据规模", data_value, data_detail),
        ("", "主模型", model_value, model_detail),
        ("", "支持输入", "单条 / 批量", "支持单条评价分析与批量导入。"),
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


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    data = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise ValueError("CSV 读取失败，请确认文件包含表头和规范的逗号分隔内容。") from exc
    raise ValueError("CSV 读取失败，请确认文件编码为 UTF-8 或 GB18030，并包含评价文本字段。")


def load_uploaded_rows(uploaded_file: Any) -> list[dict[str, str]]:
    df = read_uploaded_csv(uploaded_file)
    return normalize_dataframe(df)


def parse_expected_topics(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {
        item.strip()
        for item in re.split(r"[;；]", str(value or ""))
        if item.strip()
    }


def display_topics(topics: list[str] | set[str]) -> str:
    values = [str(topic).strip() for topic in topics if str(topic).strip()]
    return "、".join(values) if values else "未识别"


def sentiment_display(label: object) -> str:
    text = str(label or "").strip()
    return SENTIMENT_LABELS.get(text, text or "未识别")


def extract_actual_topics(result: dict[str, Any]) -> list[str]:
    topics = result.get("topics")
    if isinstance(topics, list):
        values = [str(topic).strip() for topic in topics if str(topic).strip()]
        if values:
            return values
    evidence_topics = [
        str(item.get("aspect", "")).strip()
        for item in result.get("topic_evidence", []) or []
        if isinstance(item, dict) and str(item.get("aspect", "")).strip()
    ]
    return list(dict.fromkeys(evidence_topics))


def test_case_result_row(index: int, case: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    expected_sentiment = str(case.get("expected_sentiment", "")).strip()
    actual_sentiment = str(result.get("sentiment", "")).strip()
    expected_topics = parse_expected_topics(case.get("expected_topics", ""))
    actual_topics = extract_actual_topics(result)
    actual_topic_set = set(actual_topics)
    missing_topics = sorted(expected_topics - actual_topic_set)
    sentiment_correct = expected_sentiment == actual_sentiment
    topic_correct = expected_topics.issubset(actual_topic_set)
    passed = sentiment_correct and topic_correct
    if passed:
        note = "情感与课程维度均符合预期。"
    else:
        reasons = []
        if not sentiment_correct:
            reasons.append("情感不一致")
        if not topic_correct:
            reasons.append(f"缺少预期维度：{display_topics(missing_topics)}")
        note = "；".join(reasons)

    return {
        "编号": case.get("id", index),
        "评价文本": str(case.get("text", "")).strip(),
        "预期情感": sentiment_display(expected_sentiment),
        "实际情感": sentiment_display(actual_sentiment),
        "情感正确": "是" if sentiment_correct else "否",
        "预期维度": display_topics(sorted(expected_topics)),
        "实际维度": display_topics(actual_topics),
        "维度正确": "是" if topic_correct else "否",
        "是否通过": "通过" if passed else "未通过",
        "说明": note,
    }


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


MAINTAIN_SUGGESTION_MARKERS = (
    "继续保持",
    "保持",
    "保留",
    "巩固",
    "延续",
    "推广",
    "维持",
)

IMPROVE_SUGGESTION_MARKERS = (
    "优化",
    "改进",
    "调整",
    "减少",
    "增加",
    "完善",
    "缓解",
    "降低",
    "解决",
    "延长",
    "明确",
    "补充",
    "改善",
    "减轻",
)


def advice_badge_info(suggestion: dict[str, Any], risk_level: object) -> tuple[str, str]:
    action_type = str(suggestion.get("action_type", "")).strip().lower()
    if action_type in {"maintain", "preserve", "protect", "promote", "keep"}:
        return "保持优势", "success"
    if action_type in {"priority", "urgent", "problem"}:
        return "重点问题", "accent"
    if action_type in {"improve", "optimize", "adjust"}:
        return "可优化", "warning"

    text = str(suggestion.get("suggestion", "")).strip()
    has_maintain_marker = any(marker in text for marker in MAINTAIN_SUGGESTION_MARKERS)
    has_improve_marker = any(marker in text for marker in IMPROVE_SUGGESTION_MARKERS)
    if has_improve_marker:
        return "可优化", "warning"
    if has_maintain_marker and not has_improve_marker:
        return "保持优势", "success"

    return risk_status_text(risk_level), risk_badge_class(risk_level)


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
    badge, badge_class = advice_badge_info(suggestion, risk_level)
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
        st.html(
            f"""
            <div class="advice-summary-card">
                <div class="advice-summary-icon">✦</div>
                <div class="advice-summary-content">
                    <div class="advice-summary-meta">大模型建议摘要</div>
                    <p>{escaped(advice.get("summary", "本次建议未生成总结。"))}</p>
                </div>
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
            <div class="advice-summary-card">
                <div class="advice-summary-icon">i</div>
                <div class="advice-summary-content">
                    <div class="advice-summary-meta">建议生成未启用</div>
                    <p>开启“生成改进建议”后，系统将结合评价内容给出可复核的教学建议。</p>
                </div>
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

    frame = frame.copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce").fillna(0)

    if color == CHART_COLORS["accent"]:
        color_range = ["#EFE8FF", "#B89AF8", CHART_COLORS["accent"]]
    else:
        color_range = ["#EFE8FF", "#9C7AF1", CHART_COLORS["primary"]]

    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y(f"{x_col}:N", sort="-x", title=None),
            x=alt.X(f"{y_col}:Q", title=None),
            color=alt.Color(
                f"{y_col}:Q",
                scale=alt.Scale(range=color_range),
                legend=None,
            ),
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
    batch_metrics = [
        ("", "评论总数", count_with_unit(total, "条"), f"数据来源：{source_name}"),
        ("", "积极反馈占比", format_percent(positive_rate), "反映学生整体认可程度。"),
        ("", "待关注评论", count_with_unit(len(attention_results), "条"), "当前识别为消极的评价数量。"),
        ("", "主要关注维度", top_topic, "本批评价中被提及最多的教学维度。"),
    ]
    render_metric_grid(batch_metrics, "single-overview-grid")

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
                        "置信度": st.column_config.NumberColumn("置信度", format="%.2f"),
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
                "置信度": st.column_config.NumberColumn("置信度", format="%.2f"),
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
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error("CSV 读取失败，请确认文件编码为 UTF-8 或 GB18030，并包含评价文本字段。")
            return
    else:
        rows = st.session_state.get("batch_sample_rows", [])
        source_name = st.session_state.get("batch_sample_source", "")

    if not rows:
        render_notice("等待数据", "请上传 CSV，或点击“加载项目已有示例数据”。")
        return

    preview = pd.DataFrame(rows)
    with st.container(border=True):
        render_card_title("数据预览", f"共 {len(preview)} 条记录，开始分析前可检查文本字段和课程信息。")
        preview_height = min(430, max(220, 64 + min(len(preview), 10) * 36))
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            height=preview_height,
            column_config={
                column: st.column_config.TextColumn(column, width="large")
                for column in ("review_text", "text")
                if column in preview.columns
            },
        )

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


def test_case_source_key(path: Path, frame: pd.DataFrame) -> str:
    version = path.stat().st_mtime_ns if path.exists() else 0
    return f"{version}:{len(frame)}"


def page_test_cases() -> None:
    render_page_header(
        "✅",
        "固定案例验证",
        "运行 12 条固定课程评价案例，检查当前 NLP 流程在典型业务场景下的情感分类和课程维度识别结果。",
    )

    if not TEST_CASES.exists():
        render_notice("未找到固定案例文件", "请确认 data/test_cases.csv 存在后重试。", "danger")
        return

    try:
        cases = load_test_case_frame(str(TEST_CASES))
    except Exception:
        st.error("固定案例文件读取失败，请确认 data/test_cases.csv 为 UTF-8 CSV，并包含表头行。")
        return

    required = {"text", "expected_sentiment", "expected_topics"}
    missing = sorted(required - set(cases.columns))
    if missing:
        st.error(f"固定案例文件缺少字段：{', '.join(missing)}。请补齐后重试。")
        return

    preview_columns = ["text", "expected_sentiment", "expected_topics"]
    preview = cases[preview_columns].rename(
        columns={
            "text": "评价文本",
            "expected_sentiment": "预期情感",
            "expected_topics": "预期维度",
        }
    )
    with st.container(border=True):
        render_card_title("测试用例预览", "固定案例覆盖中英文评价、混合表达和典型课程维度。")
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            column_config={"评价文本": st.column_config.TextColumn("评价文本", width="large")},
        )

    render_notice(
        "验证说明",
        "固定案例用于验证演示流程和关键业务场景，不等同于完整泛化能力评估。",
    )

    source_key = test_case_source_key(TEST_CASES, cases)
    if st.button("运行固定案例验证", type="primary", width="stretch"):
        rows: list[dict[str, object]] = []
        with st.spinner("正在运行固定案例验证，请稍候..."):
            for index, raw_case in cases.reset_index(drop=True).iterrows():
                case = raw_case.to_dict()
                text = str(case.get("text", "")).strip()
                if not text:
                    continue
                result = run_single_analysis(text, use_llm=False)
                rows.append(test_case_result_row(index + 1, case, result))
        st.session_state["test_case_results"] = rows
        st.session_state["test_case_source_key"] = source_key
        st.rerun()

    results = st.session_state.get("test_case_results")
    if st.session_state.get("test_case_source_key") != source_key or not results:
        return

    result_frame = pd.DataFrame(results)
    total = len(result_frame)
    passed = int((result_frame["是否通过"] == "通过").sum()) if total else 0
    failed = total - passed
    pass_rate = passed / total if total else 0

    render_section_title("验证结果")
    cards = [
        ("", "固定案例总数", count_with_unit(total, "条"), "本次参与验证的固定案例数量。"),
        ("", "通过案例", count_with_unit(passed, "条"), "情感与课程维度均符合预期。"),
        ("", "失败案例", count_with_unit(failed, "条"), "需要复核的案例数量。"),
        ("", "通过率", format_percent(pass_rate), "固定案例验证通过比例。"),
    ]
    render_metric_grid(cards, "single-overview-grid")

    with st.container(border=True):
        render_card_title("验证明细", "逐条展示预期结果、实际结果和判定说明。")
        st.dataframe(
            result_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "评价文本": st.column_config.TextColumn("评价文本", width="large"),
                "说明": st.column_config.TextColumn("说明", width="medium"),
            },
        )
        st.download_button(
            "下载验证结果 CSV",
            data=result_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="courseinsight_test_case_results.csv",
            mime="text/csv",
            width="stretch",
        )

    if failed == 0:
        render_notice(
            "固定案例全部通过",
            "固定案例全部通过。该结果用于验证演示流程，泛化能力仍以独立测试集和压力测试为准。",
            "success",
        )
    else:
        render_section_title("未通过案例")
        failed_frame = result_frame[result_frame["是否通过"] != "通过"]
        st.dataframe(
            failed_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "评价文本": st.column_config.TextColumn("评价文本", width="large"),
                "说明": st.column_config.TextColumn("说明", width="medium"),
            },
        )


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
                "模型类型": "最终实验主模型",
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
    if not matrix:
        return pd.DataFrame()
    labels = metrics.get("labels", [])
    if not labels and len(matrix) == len(DEFAULT_SENTIMENT_LABEL_ORDER):
        labels = list(DEFAULT_SENTIMENT_LABEL_ORDER)
    if not labels:
        return pd.DataFrame()
    readable_labels = [SENTIMENT_LABELS.get(str(label), str(label)) for label in labels]
    return pd.DataFrame(matrix, index=readable_labels, columns=readable_labels)


def blue_header_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> Any:
    styler = frame.style.hide(axis="index").set_properties(
        **{
            "background-color": "#FFFFFF",
            "color": "#12284A",
            "border-color": "#D9E8F8",
            "font-weight": "500",
        }
    )
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#174F8A"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "700"),
                    ("border-color", "#174F8A"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-color", "#D9E8F8"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [("background-color", "#F8FBFF")],
            },
        ],
        overwrite=False,
    )
    return styler.format(formats or {}, na_rep="待补充")


def confusion_matrix_chart(matrix_frame: pd.DataFrame) -> alt.Chart:
    if matrix_frame.empty:
        return alt.Chart(pd.DataFrame(columns=["真实类别", "预测类别", "数量"])).mark_rect()

    display_frame = matrix_frame.reset_index().rename(columns={"index": "真实类别"})
    long_frame = display_frame.melt(
        id_vars="真实类别",
        var_name="预测类别",
        value_name="数量",
    )
    long_frame["数量"] = pd.to_numeric(long_frame["数量"], errors="coerce").fillna(0)
    label_order = list(matrix_frame.index)
    max_value = float(long_frame["数量"].max()) if not long_frame.empty else 0.0
    text_threshold = max_value * 0.58

    base = alt.Chart(long_frame).encode(
        x=alt.X(
            "预测类别:N",
            title=None,
            sort=list(matrix_frame.columns),
            axis=alt.Axis(
                orient="top",
                labelAngle=0,
                labelPadding=12,
                labelColor=CHART_COLORS["muted"],
                domain=False,
                ticks=False,
            ),
        ),
        y=alt.Y(
            "真实类别:N",
            title=None,
            sort=label_order,
            axis=alt.Axis(
                labelPadding=12,
                labelColor=CHART_COLORS["muted"],
                domain=False,
                ticks=False,
            ),
        ),
    )
    heatmap = base.mark_rect(cornerRadius=6, stroke="#EAF3FC", strokeWidth=2).encode(
        color=alt.Color(
            "数量:Q",
            scale=alt.Scale(
                range=[
                    CHART_COLORS["bert_blue_pale"],
                    CHART_COLORS["bert_blue_soft"],
                    CHART_COLORS["bert_blue"],
                    CHART_COLORS["bert_blue_dark"],
                ]
            ),
            legend=alt.Legend(
                title="样本数",
                orient="right",
                labelColor=CHART_COLORS["muted"],
                titleColor=CHART_COLORS["muted"],
            ),
        ),
        tooltip=["真实类别:N", "预测类别:N", alt.Tooltip("数量:Q", format=".0f")],
    )
    labels = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text=alt.Text("数量:Q", format=".0f"),
        color=alt.condition(
            alt.datum.数量 > text_threshold,
            alt.value("#FFFFFF"),
            alt.value("#182236"),
        ),
    )
    return (
        (heatmap + labels)
        .properties(height=320)
        .configure_axis(labelFontSize=13, labelColor=CHART_COLORS["muted"])
        .configure_legend(labelFontSize=12, titleFontSize=13)
        .configure_view(strokeWidth=0)
    )


def model_metric_chart(frame: pd.DataFrame) -> alt.Chart:
    if frame.empty:
        return alt.Chart(pd.DataFrame(columns=["模型", "指标", "分数"])).mark_bar()
    long_frame = frame.melt(
        id_vars=["模型"],
        value_vars=["Accuracy", "Macro-F1"],
        var_name="指标",
        value_name="分数",
    ).dropna()
    long_frame["分数"] = pd.to_numeric(long_frame["分数"], errors="coerce")
    long_frame = long_frame.dropna(subset=["分数"])
    long_frame["分数标签"] = long_frame["分数"].map(lambda value: f"{value:.1%}")
    max_score = float(long_frame["分数"].max()) if not long_frame.empty else 1.0
    x_upper = min(1.0, max(0.4, round((max_score + 0.06) * 20 + 0.499) / 20))
    metric_order = ["Accuracy", "Macro-F1"]
    model_order = list(frame["模型"])

    base = alt.Chart(long_frame).encode(
        y=alt.Y(
            "模型:N",
            sort=model_order,
            title=None,
            axis=alt.Axis(
                labelLimit=220,
                labelPadding=14,
                labelColor=CHART_COLORS["muted"],
                domain=False,
                ticks=False,
            ),
        ),
        x=alt.X(
            "分数:Q",
            title=None,
            scale=alt.Scale(domain=[0, x_upper], nice=False),
            axis=alt.Axis(
                format=".0%",
                tickCount=5,
                grid=True,
                gridColor="#D9E8F8",
                labelColor=CHART_COLORS["muted"],
                domain=False,
                ticks=False,
            ),
        ),
        yOffset=alt.YOffset("指标:N", sort=metric_order),
        tooltip=["模型:N", "指标:N", alt.Tooltip("分数:Q", format=".4f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=6, size=11).encode(
        color=alt.Color(
            "指标:N",
            scale=alt.Scale(
                domain=metric_order,
                range=[CHART_COLORS["bert_blue"], CHART_COLORS["bert_blue_soft"]],
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
                direction="horizontal",
                symbolType="square",
                labelColor=CHART_COLORS["muted"],
            ),
        ),
    )
    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        fontSize=12,
        fontWeight="bold",
        color=CHART_COLORS["muted"],
    ).encode(text=alt.Text("分数标签:N"))
    return (
        (bars + labels)
        .properties(height=max(240, 44 * frame["模型"].nunique()), padding={"right": 56})
        .configure_axis(labelFontSize=13, labelColor=CHART_COLORS["muted"])
        .configure_legend(labelFontSize=13, orient="top", padding=0)
        .configure_view(strokeWidth=0)
    )


def page_model_eval() -> None:
    render_page_header(
        "🧠",
        "模型评估",
        "本页展示离线实验评估结果，并同步显示当前演示环境实际调用的运行后端。",
    )

    model_name, metrics = current_eval_metrics()
    support_total = classification_support_total(metrics)
    cards = [
        ("", "最终实验主模型", model_name, "离线测试集上的主模型结果。"),
        ("", "测试集 Accuracy", format_percent(metrics.get("accuracy")), "离线测试集整体分类正确比例。"),
        ("", "Macro-F1", format_decimal(macro_f1_value(metrics)), "三类情感分类的宏平均 F1。"),
        ("", "评估样本", count_with_unit(support_total, "条") if support_total else "缺少数据", "分类报告中的测试样本数。"),
    ]
    render_metric_grid(cards, "single-overview-grid")

    runtime = load_backend()["runtime_status"]()
    bert = runtime.get("bert", {}) if isinstance(runtime.get("bert"), dict) else {}
    render_section_title("当前运行状态")
    runtime_cards = [
        ("", "配置后端", backend_display_name(runtime.get("configured_backend")), "来自 SENTIMENT_BACKEND。"),
        ("", "实际运行后端", backend_display_name(runtime.get("active_backend")), "本机当前推理会调用的后端。"),
        ("", "BERT 权重状态", bert_runtime_status_label(bert), bert_runtime_status_detail(bert)),
        ("", "TF-IDF 模型状态", tfidf_runtime_status_label(runtime.get("tfidf_available")), "检查模型与向量器文件。"),
        ("", "LLM 配置状态", "已配置" if llm_is_configured(current_env_version()) else "未配置", "只影响教学建议生成。"),
    ]
    render_metric_grid(runtime_cards)

    render_section_title("模型对比表")
    comparison = model_comparison_frame()
    with st.container(border=True):
        render_card_title("模型指标对比", "最终实验主模型为 multilingual BERT；传统机器学习模型作为对照。")
        if comparison.empty:
            st.info("缺少评估文件：当前未找到模型对比指标。")
        else:
            st.altair_chart(model_metric_chart(comparison), width="stretch")
            st.table(
                blue_header_table(
                    comparison,
                    {
                        "Accuracy": "{:.4f}",
                        "Macro-F1": "{:.4f}",
                    },
                )
            )

    render_section_title("混淆矩阵与分类报告")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_card_title(f"{model_name} 混淆矩阵", "展示离线评估中不同情感类别之间的识别情况。")
            current_matrix = confusion_matrix_frame(metrics)
            if current_matrix.empty:
                st.info("缺少最终评估数据：当前未找到可展示的混淆矩阵。")
            else:
                st.caption(f"{model_name} 混淆矩阵（离线评估结果）")
                st.altair_chart(confusion_matrix_chart(current_matrix), width="stretch")
    with right:
        with st.container(border=True):
            render_card_title("分类报告", "展示不同情感类别下的识别表现。")
            report_frame = classification_report_frame(metrics)
            if report_frame.empty:
                st.info("缺少评估文件：当前未找到可展示的分类报告。")
            else:
                st.table(
                    blue_header_table(
                        report_frame,
                        {
                            "Precision": "{:.4f}",
                            "Recall": "{:.4f}",
                            "F1-score": "{:.4f}",
                            "Support": "{:.0f}",
                        },
                    )
                )

    render_section_title("模型选择说明")
    cols = st.columns(3)
    explanations = [
        ("为什么选择 BERT", "multilingual BERT 更适合处理中英文混合评价和上下文表达。"),
        ("运行后端说明", "当前运行后端以本机模型文件、依赖和配置为准，会在本页展示。"),
        ("后续优化", "对于讽刺表达、隐含抱怨和更细粒度教学问题，还需要更多标注数据支持。"),
    ]
    for col, item in zip(cols, explanations):
        with col:
            render_info_card(*item)

    ablation_metrics = load_json_file(str(ABLATION_METRICS))
    if ablation_metrics.get("experiments"):
        with st.container(border=True):
            render_card_title("消融实验：组件贡献分析", "固定案例验证完整流程和各组件在典型场景下的表现。")
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
            ablation_frame = pd.DataFrame(rows)
            st.table(
                blue_header_table(
                    ablation_frame,
                    {
                        "Accuracy": "{:.4f}",
                        "Macro-F1": "{:.4f}",
                        "通过数": "{:.0f}",
                        "失败数": "{:.0f}",
                    },
                )
            )


def main() -> None:
    page = render_shell()
    if page == "🏠 首页概览":
        page_home()
    elif page == "📄 单条分析":
        page_single_analysis()
    elif page == "📊 批量分析":
        page_batch_analysis()
    elif page == "✅ 固定案例验证":
        page_test_cases()
    else:
        page_model_eval()


if __name__ == "__main__":
    main()
