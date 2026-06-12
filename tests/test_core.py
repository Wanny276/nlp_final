import unittest
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app import build_test_case_results, parse_expected_topics
from scripts.prepare_coursera_dataset import prepare_dataset
from src.keyword_extractor import keywords_only
from src import topic_analyzer
from src.llm_client import (
    LLMConfig,
    REVIEW_ADVICE_SCHEMA,
    build_single_review_prompt,
    call_llm_json,
    generate_review_advice,
    local_summary,
)
from src.nlp_analyzer import analyze_review, sentiment_distribution
from src.data_loader import label_from_rating, load_reviews_csv
from src.preprocess import clean_text, detect_language, tokenize
from src.similarity import cosine_similarity
from src.train_model import train
from src.topic_analyzer import detect_topics


class CorePipelineTest(unittest.TestCase):
    def test_clean_text_removes_noise(self):
        self.assertEqual(clean_text(" 老师讲得很好！！！ T_T "), "老师讲得很好！！！")

    def test_tokenize_keeps_meaningful_words(self):
        tokens = tokenize("实验环境配置太复杂")
        self.assertTrue(tokens)

    def test_detect_english_language(self):
        self.assertEqual(detect_language("The instructor explains concepts clearly"), "en")

    def test_english_tokenize_removes_common_stopwords(self):
        tokens = tokenize("The instructor explains concepts clearly")
        self.assertIn("explains", tokens)
        self.assertNotIn("the", tokens)

    def test_topic_detection(self):
        topics = detect_topics("实验环境配置太复杂，经常运行报错")
        self.assertIn("实验实践", topics)

    def test_english_topic_detection(self):
        topics = detect_topics("The assignments are too many and the deadline is stressful")
        self.assertIn("作业任务", topics)

    def test_example_suggestion_hits_content_topic(self):
        topics = detect_topics("希望老师能多给一些代码示例")
        self.assertIn("教学内容", topics)

    def test_english_difficult_review_hits_content_topic(self):
        topics = detect_topics("The course is difficult but I learned practical skills")
        self.assertIn("教学内容", topics)

    def test_keyword_extraction(self):
        keywords = keywords_only("作业太多了，作业提交时间也紧", top_k=5)
        self.assertIn("作业", keywords)
        self.assertIn("提交", keywords)

    def test_similarity(self):
        score = cosine_similarity("实验环境配置复杂", "实验配置步骤太麻烦")
        self.assertGreater(score, 0)

    @patch("src.nlp_analyzer.model_based_sentiment", return_value=None)
    def test_analyze_review(self, _mock_model):
        result = analyze_review("老师讲课很清楚，课堂互动很多", use_llm=False)
        self.assertEqual(result["sentiment"], "positive")
        self.assertIn("授课方式", result["topics"])

    @patch("src.nlp_analyzer.model_based_sentiment", return_value=None)
    def test_analyze_english_review(self, _mock_model):
        result = analyze_review(
            "The instructor explains concepts clearly but the assignments are too many",
            use_llm=False,
        )
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertIn("作业任务", result["topics"])

    @patch("src.nlp_analyzer.model_based_sentiment", return_value=None)
    def test_mixed_bilingual_review_is_neutral(self, _mock_model):
        result = analyze_review("老师讲解很 clear，但是 assignment 太多，deadline 有点紧。", use_llm=False)
        self.assertEqual(result["language"], "mixed")
        self.assertEqual(result["sentiment"], "neutral")

    @patch("src.nlp_analyzer.model_based_sentiment", return_value=("negative", 0.74))
    def test_balanced_mixed_review_overrides_uncertain_model(self, _mock_model):
        result = analyze_review(
            "The instructor explains concepts clearly but the assignments are too many and the setup is confusing.",
            use_llm=False,
        )
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["sentiment_source"], "hybrid")

    def test_sentiment_distribution(self):
        results = [
            {"sentiment": "positive"},
            {"sentiment": "negative"},
            {"sentiment": "positive"},
        ]
        self.assertEqual(sentiment_distribution(results)["positive"], 2)

    def test_local_llm_fallback(self):
        result = local_summary(
            {
                "sentiment": "negative",
                "topics": ["实验实践"],
                "keywords": ["实验", "配置"],
            }
        )
        self.assertEqual(result["source"], "local_fallback")
        self.assertEqual(result["risk_level"], "high")

    def test_topic_evidence_includes_keywords_and_snippet(self):
        evidence = topic_analyzer.detect_topic_evidence(
            "The assignments are too many, the deadlines are stressful, and the setup is confusing."
        )

        aspects = {item["aspect"] for item in evidence}
        self.assertIn("作业任务", aspects)
        homework = next(item for item in evidence if item["aspect"] == "作业任务")
        self.assertIn("assignments", homework["keywords"])
        self.assertIn("assignments", homework["evidence"].lower())

    def test_review_prompt_requires_actionable_evidence_bound_suggestions(self):
        prompt = build_single_review_prompt(
            {
                "text": "老师讲课逻辑清晰，案例很实用，课堂互动也很多。",
                "sentiment": "positive",
                "topics": ["授课方式", "教学内容"],
                "keywords": ["讲课", "案例", "互动"],
                "topic_evidence": [
                    {
                        "aspect": "授课方式",
                        "keywords": ["讲课", "互动"],
                        "evidence": "老师讲课逻辑清晰，课堂互动也很多。",
                    }
                ],
                "similar_reviews": [],
            }
        )

        self.assertIn("suggestions 至少 1 条", prompt)
        self.assertIn("正面评价", prompt)
        self.assertIn("aspect 必须优先来自 topic_evidence.aspect", prompt)
        self.assertIn("evidence 必须引用", prompt)
        self.assertEqual(REVIEW_ADVICE_SCHEMA["properties"]["suggestions"]["minItems"], 1)

    @patch("src.llm_client.requests.post")
    def test_call_llm_json_uses_ecnu_schema_payload(self, mock_post):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"作业压力较大",'
                                    '"problems":[{"aspect":"作业任务","description":"作业偏多","evidence":"assignments are too many"}],'
                                    '"suggestions":[{"aspect":"作业任务","suggestion":"分散截止时间","evidence":"deadlines are stressful"}],'
                                    '"risk_level":"high"}'
                                )
                            }
                        }
                    ]
                }

        mock_post.return_value = FakeResponse()
        config = LLMConfig(
            api_key="test-key",
            base_url="https://chat.ecnu.edu.cn/open/api/v1",
            model="ecnu-plus",
            timeout=7,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            retries=0,
        )

        result = call_llm_json("请分析课程评价", config=config)

        self.assertEqual(result["risk_level"], "high")
        url = mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(url, "https://chat.ecnu.edu.cn/open/api/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(payload["model"], "ecnu-plus")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")

    @patch("src.llm_client.requests.post")
    def test_generate_review_advice_retries_then_falls_back(self, mock_post):
        mock_post.side_effect = RuntimeError("network unavailable")
        config = LLMConfig(api_key="test-key", retries=2, timeout=1)

        result = generate_review_advice(
            {
                "text": "实验环境配置太复杂，经常报错",
                "sentiment": "negative",
                "topics": ["实验实践"],
                "keywords": ["实验", "配置"],
                "topic_evidence": [
                    {
                        "aspect": "实验实践",
                        "keywords": ["实验", "配置"],
                        "evidence": "实验环境配置太复杂",
                    }
                ],
                "similar_reviews": [{"text": "实验配置步骤太麻烦", "score": 0.8}],
            },
            config=config,
        )

        self.assertEqual(result["source"], "local_fallback")
        self.assertEqual(mock_post.call_count, 3)

    def test_rating_to_label(self):
        self.assertEqual(label_from_rating("5"), "positive")
        self.assertEqual(label_from_rating("3"), "neutral")
        self.assertEqual(label_from_rating("1"), "negative")

    def test_load_coursera_like_csv(self):
        rows = load_reviews_csv("data/coursera_sample_reviews.csv")
        self.assertEqual(rows[0]["text"], "The instructor explains every concept clearly and the examples are useful")
        self.assertEqual(rows[0]["label"], "positive")

    def test_prepare_coursera_dataset_filters_by_text_length(self):
        input_path = Path("tests/fixtures/coursera_prepare_raw.csv")
        output_path = Path("outputs/reports/test_prepare_coursera_dataset.csv")

        sampled = prepare_dataset(
            input_path=input_path,
            output_path=output_path,
            per_label=2,
            min_chars=20,
            max_chars=60,
        )

        self.assertEqual(len(sampled), 3)
        self.assertEqual(set(sampled["label"]), {"negative", "neutral", "positive"})
        self.assertTrue(output_path.exists())

    def test_parse_expected_topics(self):
        self.assertEqual(parse_expected_topics("教学内容;考试安排；学习收获"), ["教学内容", "考试安排", "学习收获"])

    @patch("app.analyze_review")
    def test_build_test_case_results(self, mock_analyze):
        mock_analyze.return_value = {
            "language": "zh",
            "sentiment": "positive",
            "confidence": 0.91,
            "sentiment_source": "rule",
            "topics": ["授课方式", "教学内容"],
            "keywords": ["清楚", "内容"],
        }
        cases = pd.DataFrame(
            [
                {
                    "id": 1,
                    "text": "老师讲课很清楚，内容也很有用",
                    "expected_sentiment": "positive",
                    "expected_topics": "授课方式;教学内容",
                    "note": "正面评价",
                }
            ]
        )

        results = build_test_case_results(cases)

        self.assertEqual(results.loc[0, "是否通过"], "通过")
        self.assertEqual(results.loc[0, "情感来源"], "规则兜底")
        self.assertEqual(results.loc[0, "关键词"], "清楚、内容")

    def test_train_writes_reports_charts_and_language_metrics(self):
        rows = []
        examples = {
            "positive": [
                ("zh", "老师讲解清楚，内容很有帮助"),
                ("en", "The instructor explains clearly and the course is helpful"),
            ],
            "neutral": [
                ("zh", "课程内容有用，但是作业有点多"),
                ("en", "The course is useful but the assignments are many"),
            ],
            "negative": [
                ("zh", "实验环境复杂，经常报错"),
                ("en", "The setup is confusing and the assignments are stressful"),
            ],
        }
        for label, samples in examples.items():
            for index in range(6):
                language, text = samples[index % 2]
                rows.append(
                    {
                        "text": f"{text} {index}",
                        "label": label,
                        "language": language,
                    }
                )

        root = Path("outputs/reports") / f"test_train_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            data_path = root / "train.csv"
            model_dir = root / "models"
            report_dir = root / "reports"
            chart_dir = root / "charts"
            pd.DataFrame(rows).to_csv(data_path, index=False, encoding="utf-8")

            metrics = train(
                data_path,
                model_dir=model_dir,
                report_dir=report_dir,
                chart_dir=chart_dir,
            )

            self.assertTrue((model_dir / "model_metrics.json").exists())
            self.assertTrue((report_dir / "classification_report.json").exists())
            self.assertTrue((chart_dir / "confusion_matrix.png").exists())
            self.assertTrue((chart_dir / "model_comparison.png").exists())
            self.assertIn("classification_report", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("overall", metrics["language_metrics"])
            self.assertIn("en", metrics["language_metrics"])
            self.assertIn("zh", metrics["language_metrics"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
