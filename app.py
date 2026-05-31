"""Streamlit entry point for CourseInsight."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_loader import load_reviews_csv, rows_to_texts
from src.keyword_extractor import extract_keywords
from src.nlp_analyzer import analyze_batch, analyze_review, sentiment_distribution, topic_distribution


SAMPLE_DATA = Path("data/sample_reviews.csv")
SENTIMENT_LABELS = {
    "positive": "正面 positive",
    "neutral": "中性 neutral",
    "negative": "负面 negative",
}


def load_reference_reviews() -> list[str]:
    if not SAMPLE_DATA.exists():
        return []
    return rows_to_texts(load_reviews_csv(SAMPLE_DATA))


def render_single_review_page() -> None:
    st.subheader("单条评价分析 / Single Review Analysis")
    samples = {
        "中文示例": "老师讲得很清楚，但是作业有点多，实验环境配置也比较麻烦。",
        "English example": "The instructor explains concepts clearly, but the assignments are too many and the setup is confusing.",
    }
    sample_key = st.radio("示例语言", list(samples), horizontal=True)
    sample = samples[sample_key]
    text = st.text_area("输入课程评价", value=sample, height=120)
    use_llm = st.checkbox("生成大模型总结/本地兜底建议", value=True)

    if st.button("开始分析", type="primary"):
        result = analyze_review(text, reference_reviews=load_reference_reviews(), use_llm=use_llm)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("语言", result["language"])
        col_b.metric("置信度", result["confidence"])
        col_c.metric("情感倾向", SENTIMENT_LABELS.get(result["sentiment"], result["sentiment"]))
        col_d.metric("主题数量", len(result["topics"]))

        st.write("主题：", "、".join(result["topics"]) or "未识别")
        st.write("关键词：", "、".join(result["keywords"]) or "无")
        st.caption(f"预处理结果：{result['processed_text']}")

        if result.get("llm_advice"):
            st.markdown("#### 总结与建议")
            st.json(result["llm_advice"])

        if result["similar_reviews"]:
            st.markdown("#### 相似评论")
            st.dataframe(result["similar_reviews"], use_container_width=True)


def render_batch_page() -> None:
    st.subheader("批量 CSV 分析 / Batch CSV Analysis")
    st.caption("支持 text/review/reviews/comment/content 字段；如包含 rating 字段，会自动转换为情感标签。")
    uploaded = st.file_uploader("上传课程评论 CSV 文件", type=["csv"])

    if uploaded is None:
        rows = load_reviews_csv(SAMPLE_DATA)
        st.info("当前展示示例数据。上传 CSV 后会自动替换。")
    else:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(uploaded.getvalue())
            temp_path = temp_file.name
        rows = load_reviews_csv(temp_path)

    texts = rows_to_texts(rows)
    st.write(f"共读取 {len(texts)} 条有效评价")

    if not texts:
        st.warning("没有可分析的文本。")
        return

    if st.button("分析批量数据", type="primary"):
        results = analyze_batch(texts, use_llm=False)
        sentiments = sentiment_distribution(results)
        topics = topic_distribution(results)
        keywords = dict(extract_keywords(texts, top_k=12))

        col_a, col_b = st.columns(2)
        col_a.markdown("#### 情感分布")
        col_a.bar_chart(sentiments)
        col_b.markdown("#### 主题分布")
        col_b.bar_chart(topics)

        st.markdown("#### 高频关键词")
        st.bar_chart(keywords)

        st.markdown("#### 明细结果")
        st.dataframe(
            [
                {
                    "评价文本": item["text"],
                    "语言": item["language"],
                    "情感": SENTIMENT_LABELS.get(item["sentiment"], item["sentiment"]),
                    "置信度": item["confidence"],
                    "主题": "、".join(item["topics"]),
                    "关键词": "、".join(item["keywords"]),
                }
                for item in results
            ],
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="CourseInsight", layout="wide")
    st.title("CourseInsight 中英双语课程评价智能分析系统")
    st.markdown(
        "本系统支持中文高校课程评价和英文在线课程评论，提供情感分析、主题识别、关键词提取、"
        "相似评论检索和大模型总结建议。"
    )

    st.sidebar.markdown("### 项目功能")
    st.sidebar.markdown("- 中英双语文本预处理\n- 情感倾向分析\n- 课程主题识别\n- 批量 CSV 可视化\n- LLM 总结与建议")

    page = st.sidebar.radio("功能", ["单条评价分析", "批量 CSV 分析"])
    if page == "单条评价分析":
        render_single_review_page()
    else:
        render_batch_page()


if __name__ == "__main__":
    main()
