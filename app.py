"""CourseInsight 的 Streamlit 入口。"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_reviews_csv, rows_to_texts
from src.keyword_extractor import extract_keywords
from src.nlp_analyzer import analyze_batch, analyze_review, sentiment_distribution, topic_distribution


SAMPLE_DATA = Path("data/sample_reviews.csv")
TEST_CASES = Path("data/test_cases.csv")
COURSERA_SAMPLE = Path("data/coursera_sample_reviews.csv")
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


def load_reference_reviews() -> list[str]:
    if not SAMPLE_DATA.exists():
        return []
    return rows_to_texts(load_reviews_csv(SAMPLE_DATA))


def rows_to_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["text", "course", "teacher", "label"])
    return pd.DataFrame(rows)


def chart_frame(data: dict[str, int], value_name: str = "数量") -> pd.DataFrame:
    if not data:
        return pd.DataFrame({value_name: []})
    return pd.DataFrame({value_name: data}).sort_values(value_name, ascending=False)


def load_json_file(path: str | Path) -> dict:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _format_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return ""


def bert_summary_rows(metrics: dict) -> list[dict[str, object]]:
    if not metrics:
        return []
    return [
        {
            "模型": "BERT",
            "预训练模型": metrics.get("model_name", ""),
            "准确率": _format_metric(metrics.get("accuracy")),
            "宏平均F1": _format_metric(metrics.get("macro_f1")),
            "测试集": metrics.get("test_size", ""),
        }
    ]


def ablation_summary_rows(metrics: dict) -> list[dict[str, object]]:
    experiments = metrics.get("experiments", {}) if metrics else {}
    rows: list[dict[str, object]] = []
    for name, values in experiments.items():
        rows.append(
            {
                "实验版本": name,
                "状态": values.get("status", ""),
                "准确率": _format_metric(values.get("accuracy")),
                "宏平均F1": _format_metric(values.get("macro_f1")),
                "通过": values.get("passed", 0),
                "失败": values.get("failed", 0),
                "跳过": values.get("skipped", 0),
            }
        )
    return rows


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
    st.set_page_config(page_title="CourseInsight", layout="wide")
    st.title("CourseInsight 中英双语课程评价智能分析系统")
    st.markdown(
        "面向中文高校课程评价和英文在线课程评论，系统提供情感分析、主题识别、关键词提取、"
        "相似评论检索、批量可视化和大模型总结建议。"
    )

    st.sidebar.markdown("### CourseInsight")
    st.sidebar.caption("NLP + LLM 课程反馈分析")
    return st.sidebar.radio(
        "页面",
        ["项目概览", "单条评价分析", "批量 CSV 分析", "测试用例展示", "模型与技术说明"],
    )


def render_overview_page() -> None:
    st.subheader("项目概览")

    rows = load_reviews_csv(SAMPLE_DATA) if SAMPLE_DATA.exists() else []
    texts = rows_to_texts(rows)
    preview_results = analyze_batch(texts, use_llm=False) if texts else []
    sentiments = sentiment_distribution(preview_results)
    topics = topic_distribution(preview_results)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("内置样本", len(texts))
    col_b.metric("情感类别", len([value for value in sentiments.values() if value > 0]))
    col_c.metric("识别主题", len(topics))
    col_d.metric("测试用例", len(pd.read_csv(TEST_CASES)) if TEST_CASES.exists() else 0)

    st.markdown("#### 核心流程")
    st.code(
        "课程评价文本 -> 语言识别 -> 文本预处理 -> 情感/主题/关键词/相似度分析 -> LLM 总结建议 -> 图表展示",
        language="text",
    )

    st.markdown("#### 功能模块")
    col_1, col_2, col_3 = st.columns(3)
    col_1.markdown("**中英双语预处理**\n\n中文使用 jieba 分词，英文使用正则分词和停用词过滤。")
    col_2.markdown("**传统 NLP 分析**\n\n输出情感倾向、主题类别、关键词和相似评论。")
    col_3.markdown("**大模型增强**\n\n基于结构化分析结果生成反馈总结和改进建议。")

    st.markdown("#### 示例数据预览")
    st.dataframe(rows_to_frame(rows).head(8), width="stretch")

    if COURSERA_SAMPLE.exists():
        st.markdown("#### Coursera 格式样例")
        st.dataframe(rows_to_frame(load_reviews_csv(COURSERA_SAMPLE)).head(5), width="stretch")


def render_single_review_page() -> None:
    st.subheader("单条评价分析")

    samples = {
        "中文评价": "老师讲得很清楚，但是作业有点多，实验环境配置也比较麻烦。",
        "英文评价": "The instructor explains concepts clearly but the assignments are too many and the setup is confusing.",
        "中英混合": "老师讲解很 clear，但是 assignment 太多，deadline 有点紧。",
    }
    sample_key = st.segmented_control("示例", list(samples), default="中文评价")
    text = st.text_area("评价文本", value=samples[sample_key], height=130)

    col_a, col_b = st.columns([1, 2])
    use_llm = col_a.toggle("生成总结建议", value=True)
    analyze_clicked = col_b.button("开始分析", type="primary", width="stretch")

    if not analyze_clicked:
        return

    if not text.strip():
        st.warning("请输入评价文本。")
        return

    result = analyze_review(text, reference_reviews=load_reference_reviews(), use_llm=use_llm)

    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
    metric_a.metric("语言", LANGUAGE_LABELS.get(result["language"], result["language"]))
    metric_b.metric("情感倾向", SENTIMENT_LABELS.get(result["sentiment"], result["sentiment"]))
    metric_c.metric("置信度", result["confidence"])
    metric_d.metric("情感来源", SENTIMENT_SOURCE_LABELS.get(result["sentiment_source"], result["sentiment_source"]))
    metric_e.metric("主题数量", len(result["topics"]))

    tab_structured, tab_advice, tab_similar = st.tabs(["结构化结果", "总结建议", "相似评论"])

    with tab_structured:
        col_left, col_right = st.columns(2)
        col_left.markdown("#### 主题类别")
        col_left.write("、".join(result["topics"]) or "未识别")
        col_right.markdown("#### 关键词")
        col_right.write("、".join(result["keywords"]) or "无")
        if result.get("topic_evidence"):
            evidence_rows = [
                {
                    "课程维度": item["aspect"],
                    "命中关键词": "、".join(item["keywords"]),
                    "证据片段": item["evidence"],
                }
                for item in result["topic_evidence"]
            ]
            st.markdown("#### 主题证据")
            st.dataframe(pd.DataFrame(evidence_rows), width="stretch")
        st.markdown("#### 预处理结果")
        st.code(result["processed_text"] or "无", language="text")

    with tab_advice:
        if result.get("llm_advice"):
            st.json(result["llm_advice"])
        else:
            st.info("当前未启用总结建议。")

    with tab_similar:
        if result["similar_reviews"]:
            st.dataframe(result["similar_reviews"], width="stretch")
        else:
            st.info("暂无相似评论。")


def load_uploaded_or_sample_rows(uploaded_file) -> tuple[list[dict[str, str]], str]:
    if uploaded_file is None:
        return load_reviews_csv(SAMPLE_DATA), "内置样本数据"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    return load_reviews_csv(temp_path), uploaded_file.name


def render_batch_page() -> None:
    st.subheader("批量 CSV 分析")
    st.caption("兼容 text、review、reviews、comment、content 字段；rating 字段可自动转换为情感标签。")

    uploaded = st.file_uploader("课程评论 CSV", type=["csv"])
    try:
        rows, source_name = load_uploaded_or_sample_rows(uploaded)
    except Exception as exc:
        st.error(f"CSV 读取失败：{exc}")
        return

    texts = rows_to_texts(rows)
    st.markdown(f"#### 数据来源：{source_name}")
    st.dataframe(rows_to_frame(rows).head(12), width="stretch")

    col_a, col_b = st.columns([1, 3])
    col_a.metric("有效评价数", len(texts))
    reanalyze_clicked = col_b.button("重新分析批量数据", type="primary", width="stretch")

    if not texts:
        st.warning("没有可分析的文本。")
        return

    dataset_key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    should_analyze = (
        reanalyze_clicked
        or st.session_state.get("batch_dataset_key") != dataset_key
        or "batch_results" not in st.session_state
    )
    if should_analyze:
        with st.spinner("正在分析批量评价..."):
            st.session_state["batch_results"] = analyze_batch(texts, use_llm=False)
            st.session_state["batch_dataset_key"] = dataset_key

    results = st.session_state.get("batch_results")
    if not results:
        st.warning("暂无分析结果。")
        return

    sentiments = sentiment_distribution(results)
    topics = topic_distribution(results)
    keywords = dict(extract_keywords(texts, top_k=12))

    st.markdown("#### 批量分析结果")
    chart_a, chart_b = st.columns(2)
    chart_a.markdown("#### 情感分布")
    chart_a.bar_chart(chart_frame(sentiments))
    chart_b.markdown("#### 主题分布")
    if topics:
        chart_b.bar_chart(chart_frame(topics))
    else:
        chart_b.info("暂未识别到主题。")

    st.markdown("#### 高频关键词")
    if keywords:
        st.bar_chart(chart_frame(keywords, value_name="频次"))
    else:
        st.info("暂未提取到关键词。")

    st.markdown("#### 分析明细")
    results_df = pd.DataFrame(result_rows(results))
    st.dataframe(results_df, width="stretch")
    st.download_button(
        "下载分析结果 CSV",
        data=results_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="courseinsight_batch_results.csv",
        mime="text/csv",
    )


def render_test_cases_page() -> None:
    st.subheader("测试用例展示")
    if not TEST_CASES.exists():
        st.warning("未找到测试用例文件。")
        return

    cases = pd.read_csv(TEST_CASES)
    st.dataframe(cases, width="stretch")

    if not st.button("运行测试用例", type="primary"):
        return

    with st.spinner("正在运行测试用例..."):
        output_df = build_test_case_results(cases, reference_reviews=load_reference_reviews())

    passed_count = int((output_df["是否通过"] == "通过").sum())
    pass_rate = passed_count / len(output_df) if len(output_df) else 0
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("通过数量", passed_count)
    col_b.metric("用例总数", len(output_df))
    col_c.metric("通过率", f"{pass_rate:.1%}")
    st.dataframe(output_df, width="stretch")
    st.download_button(
        "下载测试结果 CSV",
        data=output_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="courseinsight_test_case_results.csv",
        mime="text/csv",
    )


def render_tech_page() -> None:
    st.subheader("模型与技术说明")

    st.markdown("#### 技术路线")
    st.code(
        """中英双语文本
  -> 语言识别
  -> 中文 jieba 分词 / 英文正则分词
  -> 停用词过滤与 TF-IDF 特征
  -> 情感分类、主题识别、关键词提取、相似度检索
  -> 大模型 API 生成总结和改进建议""",
        language="text",
    )

    st.markdown("#### 模块对应")
    module_rows = [
        ("数据读取", "src/data_loader.py", "兼容中文样本和 Coursera 风格 CSV"),
        ("文本预处理", "src/preprocess.py", "语言识别、分词、停用词过滤"),
        ("主题识别", "src/topic_analyzer.py", "中英双语关键词规则"),
        ("关键词提取", "src/keyword_extractor.py", "高频词统计"),
        ("相似评论", "src/similarity.py", "词频向量和余弦相似度"),
        ("LLM 增强", "src/llm_client.py", "结构化 Prompt 和本地兜底"),
        ("Web 界面", "app.py", "Streamlit 交互页面"),
    ]
    st.dataframe(pd.DataFrame(module_rows, columns=["模块", "文件", "说明"]), width="stretch")

    st.markdown("#### Coursera 评分转标签")
    st.dataframe(
        pd.DataFrame(
            [
                ("4-5 分", "positive", "正面课程体验"),
                ("3 分", "neutral", "中性或混合评价"),
                ("1-2 分", "negative", "负面课程体验"),
            ],
            columns=["评分", "标签", "含义"],
        ),
        width="stretch",
    )

    st.markdown("#### 模型对比结果")
    metrics = load_json_file(MODEL_METRICS)
    if metrics:
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("最佳模型", metrics.get("best_model", "未知"))
        metric_b.metric("准确率", f"{metrics.get('accuracy', 0):.4f}")
        metric_c.metric("宏平均F1", f"{metrics.get('macro_f1', 0):.4f}")
        metric_d.metric("测试集", metrics.get("test_size", 0))

        rows = [
            {
                "模型": name,
                "准确率": f"{values['accuracy']:.4f}",
                "宏平均F1": f"{values['macro_f1']:.4f}",
            }
            for name, values in metrics.get("results", {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption(f"训练数据：{metrics.get('data_path', '')}；训练集 {metrics.get('train_size', 0)} 条。")
        st.info(
            "三分类课程评论任务需要兼顾 positive、neutral、negative 三类，因此报告中建议同时展示 "
            "Macro-F1。当前量化评估主要基于英文 Coursera 抽样数据，中文样本更多用于流程演示和输入验证。"
        )
    else:
        st.info("尚未生成模型指标。运行训练命令后会显示模型对比结果。")

    st.markdown("#### BERT 对比实验")
    bert_rows = bert_summary_rows(load_json_file(BERT_METRICS))
    if bert_rows:
        st.dataframe(pd.DataFrame(bert_rows), width="stretch")
        st.caption(
            "BERT 作为深度语义模型对比实验展示，不替代当前单条评价分析中的 hybrid 主流程。"
        )
    else:
        st.info("尚未运行 BERT 对比实验。生成 outputs/bert_metrics.json 后会显示结果。")

    st.markdown("#### 消融实验结果")
    ablation_rows = ablation_summary_rows(load_json_file(ABLATION_METRICS))
    if ablation_rows:
        st.dataframe(pd.DataFrame(ablation_rows), width="stretch")
        st.caption("LLM 只用于建议生成，不纳入基础情感分类消融。")
    else:
        st.info("尚未生成消融实验结果。运行 scripts/run_ablation_experiment.py 后会显示对比表。")


def main() -> None:
    page = render_shell()
    if page == "项目概览":
        render_overview_page()
    elif page == "单条评价分析":
        render_single_review_page()
    elif page == "批量 CSV 分析":
        render_batch_page()
    elif page == "测试用例展示":
        render_test_cases_page()
    else:
        render_tech_page()


if __name__ == "__main__":
    main()
