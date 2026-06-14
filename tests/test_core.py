import unittest
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.runtime.caching import cache_data_api

os.environ.setdefault("SENTIMENT_BACKEND", "tfidf")

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
cache_data_api._LOGGER.setLevel(logging.ERROR)

from app import (
    ABLATION_METRICS,
    APP_CSS,
    BERT_METRICS,
    PAGE_OPTIONS,
    SAMPLE_REVIEWS,
    advice_card_html,
    classification_report_frame,
    collect_dimension_scores,
    confusion_matrix_chart,
    confusion_matrix_frame,
    count_with_unit,
    display_model_name,
    find_column,
    macro_report,
    model_metric_chart,
    model_comparison_frame,
    normalize_dataframe,
    rating_summary_frame,
    result_rows,
    sentiment_score,
    validate_single_analysis_result,
)
from scripts.run_ablation_experiment import run_ablation
from scripts.prepare_coursera_dataset import prepare_dataset
from scripts.run_stress_test import (
    grouped_metrics,
    load_stress_cases,
)
from src.bert_sentiment import BertConfig, _split_token_ids, bert_status
from src.keyword_extractor import keywords_only
from src.llm_client import (
    LLMConfig,
    REVIEW_ADVICE_SCHEMA,
    build_single_review_prompt,
    call_llm_json,
    generate_review_advice,
    local_summary,
)
from src.nlp_analyzer import (
    analyze_batch,
    analyze_review,
    rule_based_sentiment,
    sentiment_distribution,
    topic_distribution,
)
from src.data_loader import label_from_rating, load_reviews_csv
from src.preprocess import clean_text, detect_language, tokenize
from src.sentiment_normalizer import (
    correct_sentiment_spelling,
    normalize_sentiment_text_with_details,
)
from src.similarity import cosine_similarity, find_similar_reviews
from src.train_bert import sample_per_label
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

    def test_bert_token_windows_overlap_and_cap_extreme_text(self):
        chunks, truncated = _split_token_ids(
            list(range(10)),
            content_limit=4,
            stride=1,
            max_chunks=8,
        )

        self.assertEqual(chunks, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]])
        self.assertFalse(truncated)

        capped_chunks, capped = _split_token_ids(
            list(range(20)),
            content_limit=4,
            stride=1,
            max_chunks=2,
        )
        self.assertEqual(capped_chunks, [[0, 1, 2, 3], [3, 4, 5, 6]])
        self.assertTrue(capped)

    def test_sentiment_normalizer_expands_abbreviations_and_typos(self):
        result = normalize_sentiment_text_with_details(
            "Tbh, this crs is gud but kinda dificult."
        )

        self.assertEqual(
            result.text,
            "To be honest, this course is good but somewhat difficult.",
        )
        self.assertEqual(
            [(item.original, item.replacement, item.kind) for item in result.replacements],
            [
                ("Tbh", "To be honest", "abbreviation"),
                ("crs", "course", "abbreviation"),
                ("gud", "good", "spelling"),
                ("kinda", "somewhat", "abbreviation"),
                ("dificult", "difficult", "spelling"),
            ],
        )
        self.assertEqual(
            normalize_sentiment_text_with_details(
                "I can't recomend this crs"
            ).text,
            "I can not recommend this course",
        )

    def test_sentiment_spelling_correction_is_conservative(self):
        expected = {
            "confusin": "confusing",
            "excelent": "excellent",
            "borng": "boring",
            "goooood": "good",
            "baaaad": "bad",
        }
        for token, correction in expected.items():
            with self.subTest(token=token):
                self.assertEqual(correct_sentiment_spelling(token), correction)

        for token in [
            "clean",
            "pytorch",
            "numpy",
            "course",
            "teacher",
            "organize",
            "recommends",
        ]:
            with self.subTest(token=token):
                self.assertIsNone(correct_sentiment_spelling(token))

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

    def test_learning_outcome_topic_recognizes_mastery(self):
        topics = detect_topics("至少让我掌握了基本方法")
        self.assertIn("学习收获", topics)

    def test_keyword_extraction(self):
        keywords = keywords_only("作业太多了，作业提交时间也紧", top_k=5)
        self.assertIn("作业", keywords)
        self.assertIn("提交", keywords)

    def test_keyword_extraction_removes_function_words(self):
        keywords = keywords_only("这门课不能说没有价值，至少让我掌握了基本方法", top_k=6)
        self.assertIn("价值", keywords)
        self.assertIn("掌握", keywords)
        self.assertNotIn("不能", keywords)
        self.assertNotIn("没有", keywords)

    def test_similarity(self):
        score = cosine_similarity("实验环境配置复杂", "实验配置步骤太麻烦")
        self.assertGreater(score, 0)

    def test_similar_reviews_excludes_weak_matches(self):
        reviews = [
            "作业太多了，每周都做不完，希望能适当减少。",
            "如果能提前公布复习重点就更好了。",
        ]
        result = find_similar_reviews(
            "这门课不能说没有价值，至少让我掌握了基本方法",
            reviews,
        )
        self.assertEqual(result, [])

    @patch("src.nlp_analyzer.bert_based_sentiment", return_value=None)
    @patch("src.nlp_analyzer.model_based_sentiment", return_value=None)
    def test_analyze_review(self, _mock_model, _mock_bert):
        result = analyze_review("老师讲课很清楚，课堂互动很多", use_llm=False)
        self.assertEqual(result["sentiment"], "positive")
        self.assertIn("授课方式", result["topics"])
        self.assertEqual(result["topic_evidence"][0]["aspect"], "授课方式")

    @patch("src.nlp_analyzer.bert_based_sentiment", return_value=None)
    @patch("src.nlp_analyzer.model_based_sentiment", return_value=None)
    def test_analyze_english_review(self, _mock_model, _mock_bert):
        result = analyze_review(
            "The instructor explains concepts clearly but the assignments are too many",
            use_llm=False,
        )
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertIn("作业任务", result["topics"])

    @patch("src.nlp_analyzer.bert_based_sentiment", return_value=None)
    @patch("src.nlp_analyzer.model_based_sentiment", return_value=None)
    def test_mixed_bilingual_review_is_neutral(self, _mock_model, _mock_bert):
        result = analyze_review("老师讲解很 clear，但是 assignment 太多，deadline 有点紧。", use_llm=False)
        self.assertEqual(result["language"], "mixed")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(len(result["topics"]), len(set(result["topics"])))

    @patch("src.nlp_analyzer.bert_based_sentiment", return_value=None)
    @patch("src.nlp_analyzer.model_based_sentiment", return_value=("negative", 0.74))
    def test_balanced_mixed_review_overrides_uncertain_model(self, _mock_model, _mock_bert):
        result = analyze_review(
            "The instructor explains concepts clearly but the assignments are too many and the setup is confusing.",
            use_llm=False,
        )
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["sentiment_source"], "tfidf+rule")

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch("src.nlp_analyzer.model_based_sentiment", return_value=("negative", 0.99))
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("positive", 0.92, "cuda"),
    )
    def test_bert_is_preferred_over_tfidf(self, _mock_bert, mock_tfidf):
        result = analyze_review("The instructor is clear and helpful", use_llm=False)

        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["sentiment_source"], "bert")
        self.assertEqual(result["sentiment_device"], "cuda")
        mock_tfidf.assert_not_called()

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("positive", 0.88, "cuda", 3, 380, False),
    )
    def test_long_text_metadata_reaches_analysis_result(self, _mock_bert):
        result = analyze_review(
            "The lectures cover many useful examples across the semester.",
            use_llm=False,
        )

        self.assertEqual(result["sentiment_chunk_count"], 3)
        self.assertEqual(result["sentiment_token_count"], 380)
        self.assertTrue(result["long_text_handled"])
        self.assertFalse(result["long_text_truncated"])

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiments",
        return_value=[
            ("positive", 0.91, "cuda", 1, 12, False),
            ("negative", 0.84, "cuda", 4, 510, True),
        ],
    )
    def test_batch_preserves_chunk_metadata_per_review(self, _mock_bert):
        results = analyze_batch(
            [
                "The examples are useful.",
                "The final part is confusing and difficult.",
            ],
            use_llm=False,
        )

        self.assertEqual(results[0]["sentiment_chunk_count"], 1)
        self.assertFalse(results[0]["long_text_handled"])
        self.assertEqual(results[1]["sentiment_chunk_count"], 4)
        self.assertEqual(results[1]["sentiment_token_count"], 510)
        self.assertTrue(results[1]["long_text_truncated"])

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("negative", 0.91, "cuda"),
    )
    def test_analyze_review_normalizes_only_sentiment_input(self, mock_bert):
        original = "Tbh this crs is confusin and awfull"

        result = analyze_review(original, use_llm=False)

        self.assertEqual(result["text"], original)
        self.assertEqual(
            result["sentiment_text"],
            "To be honest this course is confusing and awful",
        )
        self.assertTrue(result["sentiment_replacements"])
        mock_bert.assert_called_once_with(
            "To be honest this course is confusing and awful"
        )

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "tfidf"})
    @patch(
        "src.nlp_analyzer.model_based_sentiment",
        return_value=("negative", 0.82),
    )
    def test_tfidf_fallback_receives_normalized_tokens(self, mock_tfidf):
        result = analyze_review("The setup is confusin", use_llm=False)

        self.assertEqual(result["text"], "The setup is confusin")
        self.assertEqual(result["sentiment_text"], "The setup is confusing")
        mock_tfidf.assert_called_once_with("setup confusing")

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("negative", 0.85, "cuda"),
    )
    def test_bert_mixed_review_uses_rule_correction(self, _mock_bert):
        result = analyze_review(
            "The content is useful but the exam scope is unclear",
            use_llm=False,
        )

        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["sentiment_source"], "bert+rule")

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("negative", 0.43, "cuda", 3, 347, False),
    )
    def test_local_negation_does_not_override_balanced_long_review(
        self,
        _mock_bert,
    ):
        result = analyze_review(
            "课程内容很有用，讲解也很清楚，但是实验环境比较复杂，"
            "考试范围没有清楚公布，总体有价值但仍需改进。",
            use_llm=False,
        )

        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["sentiment_source"], "bert+rule")
        self.assertEqual(result["sentiment_chunk_count"], 3)

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("neutral", 0.70, "cuda"),
    )
    def test_clear_negative_word_overrides_uncertain_neutral_bert(self, _mock_bert):
        result = analyze_review("The course is hard", use_llm=False)

        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["sentiment_source"], "bert+rule")

    def test_compositional_rules_handle_double_negation(self):
        positive_cases = [
            "The course is not bad",
            "The explanation is not unclear",
            "The course is not difficult",
            "老师讲得不能说不好",
            "课程并不是不值得学习",
            "这个课程不能说没有收获",
        ]
        negative_cases = [
            "The course is not useful",
            "The explanation is not clear",
            "课程不值得学习",
        ]

        for text in positive_cases:
            with self.subTest(text=text):
                self.assertEqual(rule_based_sentiment(text)[0], "positive")
        for text in negative_cases:
            with self.subTest(text=text):
                self.assertEqual(rule_based_sentiment(text)[0], "negative")

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("positive", 0.95, "cuda"),
    )
    def test_obvious_sarcasm_overrides_positive_bert(self, _mock_bert):
        cases = [
            "Great, another impossible assignment",
            "Exactly what I needed: more homework",
            "真棒，又多了一个根本做不完的作业",
            "考试范围可真“明确”",
        ]

        for text in cases:
            with self.subTest(text=text):
                result = analyze_review(text, use_llm=False)
                self.assertEqual(result["sentiment"], "negative")
                self.assertEqual(result["sentiment_source"], "bert+rule")

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch(
        "src.nlp_analyzer.bert_based_sentiment",
        return_value=("neutral", 0.80, "cuda"),
    )
    def test_implicit_complaint_overrides_neutral_bert(self, _mock_bert):
        cases = [
            "The teacher only reads the slides",
            "I spent more time fixing the setup than learning",
            "老师上课基本都在念PPT",
            "一节课大部分时间都在处理环境问题",
        ]

        for text in cases:
            with self.subTest(text=text):
                result = analyze_review(text, use_llm=False)
                self.assertEqual(result["sentiment"], "negative")
                self.assertEqual(result["sentiment_source"], "bert+rule")

    def test_suggestion_only_review_remains_neutral(self):
        cases = [
            "It would be better with fewer assignments",
            "The deadline could have been more generous",
            "希望老师能多讲一些例子",
        ]

        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(rule_based_sentiment(text)[0], "neutral")

    @patch.dict(os.environ, {"SENTIMENT_BACKEND": "auto"})
    @patch("src.nlp_analyzer.bert_based_sentiments", return_value=None)
    @patch(
        "src.nlp_analyzer.model_based_sentiments",
        return_value=[("positive", 0.9), ("negative", 0.9)],
    )
    def test_batch_falls_back_to_vectorized_tfidf(
        self,
        mock_tfidf,
        _mock_bert,
    ):
        results = analyze_batch(
            [
                "The course is clear and useful",
                "The setup is confusing and bad",
            ],
            use_llm=False,
        )

        self.assertEqual([item["sentiment"] for item in results], ["positive", "negative"])
        self.assertEqual([item["sentiment_source"] for item in results], ["tfidf", "tfidf"])
        mock_tfidf.assert_called_once()

    def test_sentiment_distribution(self):
        results = [
            {"sentiment": "positive"},
            {"sentiment": "negative"},
            {"sentiment": "positive"},
        ]
        self.assertEqual(sentiment_distribution(results)["positive"], 2)

    def test_bert_sampling_preserves_label_column(self):
        frame = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "label": ["positive", "positive", "negative", "negative"],
            }
        )

        sampled = sample_per_label(frame, per_label=1, seed=42)

        self.assertEqual(set(sampled.columns), {"text", "label"})
        self.assertEqual(
            sampled["label"].value_counts().to_dict(),
            {"negative": 1, "positive": 1},
        )

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

    def test_advice_card_uses_suggestion_tone_for_badge(self):
        html = advice_card_html(
            {
                "aspect": "授课方式",
                "suggestion": "继续保持清晰易懂的讲解风格，这是学生认可的教学亮点。",
                "evidence": "老师讲解很 clear",
            },
            "middle",
        )

        self.assertIn("保持优势", html)
        self.assertNotIn("可优化", html)

    def test_advice_card_uses_improvement_marker_before_low_risk_badge(self):
        html = advice_card_html(
            {
                "aspect": "实验实践",
                "suggestion": "建议优化期末项目的时间规划，提前发布任务或增加中间检查点，为学生预留充足的调试与完善时间。",
                "evidence": "final project 有点赶，debug 花了不少时间",
            },
            "low",
        )

        self.assertIn("可优化", html)
        self.assertNotIn("保持优势", html)

    def test_llm_schema_and_prompt_request_suggestion_action_type(self):
        suggestion_schema = REVIEW_ADVICE_SCHEMA["properties"]["suggestions"]["items"]
        prompt = build_single_review_prompt(
            {
                "text": "final project 有点赶，debug 花了不少时间",
                "language": "mixed",
                "sentiment": "negative",
                "topics": ["实验实践"],
                "keywords": ["debug"],
                "topic_evidence": [
                    {
                        "aspect": "实验实践",
                        "evidence": "final project 有点赶，debug 花了不少时间",
                    }
                ],
            }
        )

        self.assertIn("action_type", suggestion_schema["required"])
        self.assertEqual(
            set(suggestion_schema["properties"]["action_type"]["enum"]),
            {"maintain", "improve", "priority"},
        )
        self.assertIn("action_type", prompt)
        self.assertIn("maintain", prompt)
        self.assertIn("improve", prompt)
        self.assertIn("priority", prompt)

    def test_local_fallback_marks_suggestion_action_type(self):
        cases = [
            ("positive", "maintain"),
            ("neutral", "improve"),
            ("negative", "priority"),
        ]
        for sentiment, action_type in cases:
            with self.subTest(sentiment=sentiment):
                result = local_summary(
                    {
                        "text": "final project 有点赶，debug 花了不少时间",
                        "sentiment": sentiment,
                        "language": "mixed",
                        "topics": ["实验实践"],
                        "topic_evidence": [
                            {
                                "aspect": "实验实践",
                                "evidence": "final project 有点赶，debug 花了不少时间",
                            }
                        ],
                    }
                )

                self.assertEqual(result["suggestions"][0]["action_type"], action_type)

    @patch("src.llm_client.call_llm_json")
    def test_generate_review_advice_uses_api_result(self, mock_call):
        mock_call.return_value = {
            "summary": "课堂反馈整体积极。",
            "problems": [],
            "suggestions": [
                {
                    "aspect": "教学内容",
                    "suggestion": "继续保持",
                    "evidence": "讲得清楚",
                    "action_type": "maintain",
                }
            ],
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

    @patch("src.llm_client.call_llm_json")
    def test_generate_review_advice_rejects_evidence_from_other_reviews(self, mock_call):
        mock_call.return_value = {
            "summary": "建议调整作业安排。",
            "problems": [],
                    "suggestions": [
                        {
                            "aspect": "作业任务",
                            "suggestion": "适当减少作业量。",
                            "evidence": "作业太多了，每周都做不完，希望能适当减少。",
                            "action_type": "improve",
                        }
                    ],
            "risk_level": "middle",
        }

        result = generate_review_advice(
            {
                "text": "这门课不能说没有价值，至少让我掌握了基本方法",
                "sentiment": "positive",
                "topics": ["学习收获"],
                "topic_evidence": [
                    {
                        "aspect": "学习收获",
                        "evidence": "这门课不能说没有价值，至少让我掌握了基本方法",
                    }
                ],
                "keywords": ["价值", "掌握", "方法"],
            }
        )

        self.assertEqual(result["source"], "local_fallback")
        self.assertNotEqual(result["suggestions"][0]["aspect"], "作业任务")

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
                                    '"evidence":"assignment 太多","action_type":"improve"}],'
                                    '"risk_level":"middle"}'
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

    def test_frontend_navigation_is_limited_to_four_course_demo_pages(self):
        page_names = [option[2:] for option in PAGE_OPTIONS]

        self.assertEqual(page_names, ["首页概览", "单条分析", "批量分析", "模型评估"])
        self.assertNotIn("系统设置", "".join(PAGE_OPTIONS))
        self.assertNotIn("数据管理", "".join(PAGE_OPTIONS))

    def test_detail_and_advice_cards_use_adaptive_layout(self):
        self.assertIn(
            ".detail-card-grid,\n.advice-card-grid {\n    display: grid;\n"
            "    grid-template-columns: repeat(auto-fit",
            APP_CSS,
        )

    def test_sidebar_brand_keeps_logo_background_and_subtitle_single_line(self):
        self.assertIn(".brand-sidebar .ci-logo-box {", APP_CSS)
        self.assertIn("background: linear-gradient(135deg, #F8F5FF 0%, #E8E1FF 100%);", APP_CSS)
        self.assertIn("box-shadow: 0 10px 24px rgba(5, 8, 35, 0.18)", APP_CSS)
        self.assertIn("white-space: nowrap;", APP_CSS)
        self.assertIn("letter-spacing: -0.01em;", APP_CSS)

    def test_single_analysis_examples_cover_chinese_english_and_mixed_reviews(self):
        self.assertEqual(list(SAMPLE_REVIEWS), ["中文评价", "英文评价", "中英混合"])
        self.assertIn("课堂互动", SAMPLE_REVIEWS["中文评价"])
        self.assertIn("feedback often comes late", SAMPLE_REVIEWS["英文评价"])
        self.assertIn("guidance 不够详细", SAMPLE_REVIEWS["中英混合"])

    def test_single_analysis_result_validation_requires_real_backend_fields_not_score(self):
        validate_single_analysis_result(
            {
                "sentiment": "neutral",
                "confidence": 0.72,
                "keywords": ["作业"],
                "topic_evidence": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "confidence"):
            validate_single_analysis_result(
                {
                    "sentiment": "neutral",
                    "keywords": [],
                    "topic_evidence": [],
                }
            )

    def test_normalize_dataframe_requires_review_text_and_preserves_optional_fields(self):
        frame = pd.DataFrame(
            [
                {
                    "review_text": "老师讲解清楚",
                    "course_name": "NLP",
                    "teacher": "王老师",
                    "rating": 5,
                    "date": "2026-06-14",
                },
                {
                    "review_text": "老师讲解清楚",
                    "course_name": "NLP",
                    "teacher": "王老师",
                    "rating": 5,
                    "date": "2026-06-14",
                },
                {"review_text": ""},
            ]
        )

        rows = normalize_dataframe(frame)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_text"], "老师讲解清楚")
        self.assertEqual(rows[0]["course_name"], "NLP")
        self.assertEqual(rows[0]["teacher"], "王老师")
        self.assertEqual(rows[0]["rating"], "5")
        self.assertEqual(rows[0]["date"], "2026-06-14")

    def test_normalize_dataframe_accepts_compatible_text_column_and_defaults_metadata(self):
        frame = pd.DataFrame([{"comment": "作业反馈有点慢"}])

        rows = normalize_dataframe(frame)

        self.assertEqual(rows[0]["review_text"], "作业反馈有点慢")
        self.assertEqual(rows[0]["course_name"], "未提供")
        self.assertEqual(rows[0]["teacher"], "未提供")

    def test_find_column_returns_none_when_required_text_column_missing(self):
        self.assertIsNone(find_column(["course_name", "rating"], ("review_text", "text")))
        with self.assertRaisesRegex(ValueError, "review_text"):
            normalize_dataframe(pd.DataFrame([{"course_name": "NLP"}]))

    def test_result_rows_maps_backend_fields_for_batch_table(self):
        rows = [{
            "review_text": "实验说明不够详细",
            "course_name": "NLP",
            "teacher": "李老师",
            "rating": "2",
            "date": "2026-06-14",
        }]
        results = [{
            "sentiment": "negative",
            "confidence": 0.82,
            "topics": ["实验实践"],
            "keywords": ["实验", "说明"],
            "sentiment_source": "bert",
        }]

        frame_rows = result_rows(rows, results)

        self.assertEqual(frame_rows[0]["情感"], "消极")
        self.assertEqual(frame_rows[0]["分析方法"], "BERT 模型预测")
        self.assertEqual(frame_rows[0]["课程维度"], "实验实践")
        self.assertEqual(frame_rows[0]["关键词"], "实验、说明")

    def test_topic_and_rating_summaries_use_real_returned_fields(self):
        results = [
            {"topics": ["教学内容", "授课方式"]},
            {"topics": ["教学内容"]},
        ]
        topics = topic_distribution(results)
        rating_frame = rating_summary_frame(
            [
                {"review_text": "a", "course_name": "A", "teacher": "", "rating": "2"},
                {"review_text": "b", "course_name": "A", "teacher": "", "rating": "4"},
                {"review_text": "c", "course_name": "B", "teacher": "", "rating": "1"},
            ]
        )

        self.assertEqual(topics["教学内容"], 2)
        self.assertEqual(topics["授课方式"], 1)
        self.assertEqual(rating_frame.iloc[0]["对象"], "B")

    def test_collect_dimension_scores_and_sentiment_score_do_not_invent_missing_values(self):
        self.assertEqual(collect_dimension_scores({}), {})
        self.assertEqual(sentiment_score("positive"), 1.0)
        self.assertEqual(sentiment_score("neutral"), 0.5)
        self.assertEqual(sentiment_score("negative"), 0.0)
        self.assertEqual(count_with_unit(2280, "条测试样本"), "2,280 条测试样本")

    def test_model_eval_frames_read_existing_metric_shapes(self):
        metrics = {
            "accuracy": 0.75,
            "classification_report": {
                "negative": {"precision": 0.7, "recall": 0.8, "f1-score": 0.74, "support": 10},
                "macro avg": {"precision": 0.75, "recall": 0.76, "f1-score": 0.755, "support": 30},
            },
            "labels": ["negative", "neutral"],
            "confusion_matrix": [[8, 2], [3, 7]],
        }

        self.assertEqual(macro_report(metrics)["f1-score"], 0.755)
        report = classification_report_frame(metrics)
        matrix = confusion_matrix_frame(metrics)

        self.assertEqual(report.loc[0, "类别"], "消极")
        self.assertEqual(list(matrix.index), ["消极", "中性"])
        self.assertEqual(matrix.loc["消极", "中性"], 2)

    @patch("app.load_json_file")
    def test_model_comparison_frame_combines_traditional_and_bert_metrics(self, mock_load_json):
        def fake_load(path):
            if str(path).endswith("model_metrics.json"):
                return {
                    "results": {
                        "Logistic Regression": {"accuracy": 0.69, "macro_f1": 0.68},
                    }
                }
            if str(path).endswith("bert_metrics_final.json"):
                return {"accuracy": 0.75, "macro_f1": 0.72}
            return {}

        mock_load_json.side_effect = fake_load

        frame = model_comparison_frame()

        self.assertEqual(set(frame["模型"]), {"Logistic Regression", "BERT"})
        self.assertEqual(set(frame["模型类型"]), {"对比模型", "当前主模型"})
        self.assertEqual(display_model_name("linear_svm"), "Linear SVM")
        self.assertEqual(BERT_METRICS, Path("outputs/bert_metrics_final.json"))
        self.assertEqual(
            ABLATION_METRICS,
            Path("outputs/reports/final_model/ablation_metrics.json"),
        )

    def test_model_metric_chart_uses_compact_legend_and_value_labels(self):
        chart = model_metric_chart(
            pd.DataFrame(
                [
                    {"模型": "BERT", "Accuracy": 0.746, "Macro-F1": 0.745},
                    {"模型": "Dummy Most Frequent", "Accuracy": 0.333, "Macro-F1": 0.167},
                ]
            )
        )

        spec = chart.to_dict()

        self.assertIn("layer", spec)
        self.assertEqual(spec["layer"][0]["encoding"]["color"]["legend"]["orient"], "top")
        self.assertEqual(spec["layer"][0]["encoding"]["y"]["axis"]["labelLimit"], 220)
        self.assertEqual(spec["layer"][1]["mark"]["type"], "text")
        self.assertEqual(spec["layer"][1]["encoding"]["text"]["field"], "分数标签")
        color_range = spec["layer"][0]["encoding"]["color"]["scale"]["range"]
        self.assertEqual(color_range, ["#6EA2D8", "#BBD0EA"])

    def test_confusion_matrix_chart_uses_final_bert_matrix_with_blue_scale(self):
        metrics = {
            "confusion_matrix": [[590, 160, 10], [228, 460, 72], [14, 95, 651]],
        }

        matrix = confusion_matrix_frame(metrics)
        chart = confusion_matrix_chart(matrix)
        spec = chart.to_dict()

        self.assertEqual(list(matrix.index), ["消极", "中性", "积极"])
        self.assertEqual(matrix.loc["积极", "积极"], 651)
        self.assertIn("layer", spec)
        self.assertEqual(spec["layer"][0]["encoding"]["color"]["field"], "数量")
        self.assertEqual(
            spec["layer"][0]["encoding"]["color"]["scale"]["range"],
            ["#F4F8FC", "#BBD0EA", "#6EA2D8", "#174F8A"],
        )
        self.assertEqual(spec["layer"][0]["encoding"]["x"]["axis"]["orient"], "top")

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

    def test_stress_suite_has_expected_independent_structure(self):
        cases = load_stress_cases("data/stress_test_cases.csv")

        self.assertEqual(len(cases), 48)
        self.assertEqual(cases["category"].nunique(), 8)
        self.assertTrue((cases["category"].value_counts() == 6).all())
        self.assertEqual(
            set(cases["expected_sentiment"]),
            {"negative", "neutral", "positive"},
        )

        business_cases = pd.read_csv("data/test_cases.csv")
        stress_texts = set(cases["text"].str.strip().str.casefold())
        business_texts = set(
            business_cases["text"].astype(str).str.strip().str.casefold()
        )
        self.assertFalse(stress_texts & business_texts)

    def test_stress_grouped_metrics_report_each_category(self):
        cases = pd.DataFrame(
            [
                {
                    "category": "spelling",
                    "language": "en",
                    "expected_sentiment": "positive",
                },
                {
                    "category": "spelling",
                    "language": "en",
                    "expected_sentiment": "negative",
                },
                {
                    "category": "negation",
                    "language": "zh",
                    "expected_sentiment": "neutral",
                },
            ]
        )

        metrics = grouped_metrics(
            cases,
            ["positive", "neutral", "neutral"],
            "category",
        )

        self.assertEqual(metrics["spelling"]["support"], 2)
        self.assertEqual(metrics["spelling"]["passed"], 1)
        self.assertEqual(metrics["negation"]["passed"], 1)

    def test_model_eval_helpers_keep_macro_report_and_leave_missing_confusion_empty(self):
        metrics = {
            "accuracy": 0.75,
            "macro_f1": 0.72,
            "classification_report": {
                "macro avg": {
                    "precision": 0.73,
                    "recall": 0.74,
                    "f1-score": 0.72,
                    "support": 20,
                }
            },
        }

        self.assertEqual(macro_report(metrics)["precision"], 0.73)
        report = classification_report_frame(metrics)
        self.assertEqual(report.loc[0, "类别"], "macro avg")
        self.assertEqual(report.loc[0, "Support"], 20)
        self.assertTrue(confusion_matrix_frame(metrics).empty)

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

    def test_bert_status_requires_weights_and_tokenizer(self):
        model_path = Path("outputs/reports/test_bert_status_incomplete")
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "config.json").write_text("{}", encoding="utf-8")

        status = bert_status(BertConfig(model_path=model_path))

        self.assertFalse(status["model_available"])
        self.assertFalse(status["ready"])


if __name__ == "__main__":
    unittest.main()
