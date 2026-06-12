import unittest
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.runtime.caching import cache_data_api

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
cache_data_api._LOGGER.setLevel(logging.ERROR)

from app import ablation_summary_rows, bert_summary_rows, build_test_case_results, parse_expected_topics
from scripts.run_ablation_experiment import run_ablation
from scripts.prepare_coursera_dataset import prepare_dataset
from src.keyword_extractor import keywords_only
from src.llm_client import LLMConfig, call_llm_json, generate_review_advice, local_summary
from src.nlp_analyzer import analyze_review, sentiment_distribution
from src.data_loader import label_from_rating, load_reviews_csv
from src.preprocess import clean_text, detect_language, tokenize
from src.similarity import cosine_similarity
from src.train_model import train
from src.topic_analyzer import detect_topic_evidence, detect_topics


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

    def test_topic_detection_returns_evidence(self):
        evidence = detect_topic_evidence("实验环境配置太复杂，经常运行报错")
        self.assertTrue(evidence)
        self.assertEqual(evidence[0]["aspect"], "实验实践")
        self.assertIn("实验", evidence[0]["keywords"])
        self.assertIn("实验环境配置", evidence[0]["evidence"])

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
        self.assertEqual(result["topic_evidence"][0]["aspect"], "授课方式")

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
        self.assertEqual(len(result["topics"]), len(set(result["topics"])))

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
                "text": "实验环境配置太复杂，经常报错",
                "sentiment": "negative",
                "topics": ["实验实践"],
                "topic_evidence": [
                    {
                        "aspect": "实验实践",
                        "keywords": ["实验", "配置"],
                        "evidence": "实验环境配置太复杂",
                    }
                ],
                "keywords": ["实验", "配置"],
            }
        )
        self.assertEqual(result["source"], "local_fallback")
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(result["problems"][0]["aspect"], "实验实践")
        self.assertEqual(result["problems"][0]["evidence"], "实验环境配置太复杂")
        self.assertEqual(result["suggestions"][0]["aspect"], "实验实践")

    def test_local_fallback_deduplicates_topics(self):
        result = local_summary(
            {
                "text": "老师讲解很 clear，但是 assignment 太多，deadline 有点紧。",
                "sentiment": "neutral",
                "topics": ["作业任务", "作业任务", "授课方式", "授课方式"],
                "topic_evidence": [{"aspect": "作业任务", "evidence": "assignment 太多"}],
                "keywords": ["讲解", "clear", "assignment"],
            }
        )

        self.assertNotIn("作业任务、作业任务", result["summary"])
        self.assertNotIn("授课方式、授课方式", result["suggestions"][0]["suggestion"])

    @patch("src.llm_client.call_llm_json")
    def test_generate_review_advice_uses_api_result(self, mock_call):
        mock_call.return_value = {
            "summary": "课堂反馈整体积极。",
            "problems": [],
            "suggestions": [{"aspect": "教学内容", "suggestion": "继续保持", "evidence": "讲得清楚"}],
            "risk_level": "low",
        }

        result = generate_review_advice(
            {
                "text": "老师讲得清楚",
                "sentiment": "positive",
                "topics": ["教学内容"],
                "keywords": ["清楚"],
            }
        )

        self.assertEqual(result["source"], "llm_api")
        mock_call.assert_called_once()

    @patch("src.llm_client.requests.post")
    def test_llm_request_uses_compatible_payload(self, mock_post):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"反馈中性。","problems":[],'
                                    '"suggestions":[{"aspect":"作业任务","suggestion":"适当调整作业量。",'
                                    '"evidence":"assignment 太多"}],"risk_level":"middle"}'
                                )
                            }
                        }
                    ]
                }

        mock_post.return_value = FakeResponse()

        result = call_llm_json(
            "请生成建议",
            config=LLMConfig(api_key="test-key", base_url="https://example.test", retries=0),
        )
        payload = mock_post.call_args.kwargs["json"]

        self.assertEqual(result["risk_level"], "middle")
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["temperature"], 0.2)

    def test_llm_config_loads_dotenv_file(self):
        names = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT"]
        old_env = {name: os.environ.pop(name, None) for name in names}
        old_cwd = Path.cwd()
        temp_path = old_cwd / "outputs" / "reports" / "test_llm_dotenv"

        try:
            temp_path.mkdir(parents=True, exist_ok=True)
            (temp_path / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=test-key",
                        "LLM_BASE_URL=https://example.test/open/api/v1",
                        "LLM_MODEL=test-model",
                        "LLM_TIMEOUT=7",
                    ]
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)

            config = LLMConfig.from_env()

            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.base_url, "https://example.test/open/api/v1")
            self.assertEqual(config.model, "test-model")
            self.assertEqual(config.timeout, 7)
        finally:
            os.chdir(old_cwd)
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

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
        output_path = Path("outputs/reports/test_prepare_coursera_dataset/generated.csv")

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

    def test_train_model_writes_detailed_metrics(self):
        data_path = Path("outputs/reports/test_train_metrics_data.csv")
        model_dir = Path("outputs/reports/test_train_metrics_model")
        rows = []
        examples = {
            "positive": [
                "clear useful helpful course",
                "clear useful practical examples",
                "helpful practical course clear",
                "useful helpful examples clear",
            ],
            "neutral": [
                "clear but assignments many",
                "useful but deadline stressful",
                "practical but difficult course",
                "helpful but confusing setup",
            ],
            "negative": [
                "confusing difficult boring course",
                "unclear difficult stressful assignments",
                "boring confusing outdated course",
                "unclear stressful difficult setup",
            ],
        }
        for label, texts in examples.items():
            for index, text in enumerate(texts):
                rows.append(
                    {
                        "id": f"{label}-{index}",
                        "text": text,
                        "label": label,
                        "language": "en",
                    }
                )
        pd.DataFrame(rows).to_csv(data_path, index=False)

        metrics = train(data_path, model_dir=model_dir)

        self.assertIn("classification_report", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("language_metrics", metrics)
        self.assertIn("vectorizer", metrics)
        self.assertEqual(metrics["vectorizer"]["min_df"], 2)
        self.assertTrue((model_dir / "model_metrics.json").exists())

    def test_ablation_runs_without_saved_model(self):
        output_dir = Path("outputs/reports/test_ablation")
        cases = pd.DataFrame(
            [
                {
                    "id": 1,
                    "text": "老师讲课很清楚，内容也很有用",
                    "expected_sentiment": "positive",
                },
                {
                    "id": 2,
                    "text": "作业太多了，每周都做不完",
                    "expected_sentiment": "negative",
                },
            ]
        )

        metrics = run_ablation(cases, output_dir=output_dir, model_dir="outputs/reports/missing_model")

        self.assertIn("rule-only", metrics["experiments"])
        self.assertIn("hybrid", metrics["experiments"])
        self.assertEqual(metrics["experiments"]["model-only"]["status"], "跳过")
        self.assertTrue((output_dir / "ablation_metrics.json").exists())
        self.assertTrue((output_dir / "ablation_errors.csv").exists())

    def test_ablation_script_runs_as_file(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_ablation_experiment.py",
                "--cases",
                "data/test_cases.csv",
                "--output-dir",
                "outputs/reports/test_ablation_cli",
                "--model-dir",
                "outputs/reports/missing_model",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rule-only:", result.stdout)
        self.assertIn("model-only: 跳过", result.stdout)

    def test_app_formats_bert_and_ablation_metrics(self):
        bert_metrics = {
            "model_name": "bert-base-multilingual-cased",
            "accuracy": 0.75,
            "macro_f1": 0.72,
            "test_size": 20,
        }
        ablation_metrics = {
            "experiments": {
                "rule-only": {
                    "status": "completed",
                    "accuracy": 0.5,
                    "macro_f1": 0.44,
                    "passed": 1,
                    "failed": 1,
                    "skipped": 0,
                },
                "model-only": {"status": "跳过", "skipped": 2},
            }
        }

        bert_rows = bert_summary_rows(bert_metrics)
        ablation_rows = ablation_summary_rows(ablation_metrics)

        self.assertEqual(bert_rows[0]["模型"], "BERT")
        self.assertEqual(bert_rows[0]["预训练模型"], "bert-base-multilingual-cased")
        self.assertEqual(ablation_rows[0]["实验版本"], "rule-only")
        self.assertEqual(ablation_rows[1]["状态"], "跳过")

    def test_bert_help_does_not_require_transformers(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.train_bert", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("用法:", result.stdout)
        self.assertIn("选项:", result.stdout)
        self.assertIn("--sample-per-label", result.stdout)


if __name__ == "__main__":
    unittest.main()
