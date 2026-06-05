import unittest
from unittest.mock import patch

import pandas as pd

from app import build_test_case_results, parse_expected_topics
from src.keyword_extractor import keywords_only
from src.llm_client import local_summary
from src.nlp_analyzer import analyze_review, sentiment_distribution
from src.data_loader import label_from_rating, load_reviews_csv
from src.preprocess import clean_text, detect_language, tokenize
from src.similarity import cosine_similarity
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

    def test_rating_to_label(self):
        self.assertEqual(label_from_rating("5"), "positive")
        self.assertEqual(label_from_rating("3"), "neutral")
        self.assertEqual(label_from_rating("1"), "negative")

    def test_load_coursera_like_csv(self):
        rows = load_reviews_csv("data/coursera_sample_reviews.csv")
        self.assertEqual(rows[0]["text"], "The instructor explains every concept clearly and the examples are useful")
        self.assertEqual(rows[0]["label"], "positive")

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


if __name__ == "__main__":
    unittest.main()
