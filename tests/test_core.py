import unittest

from src.keyword_extractor import keywords_only
from src.llm_client import local_summary
from src.nlp_analyzer import analyze_review, sentiment_distribution
from src.preprocess import clean_text, tokenize
from src.similarity import cosine_similarity
from src.topic_analyzer import detect_topics


class CorePipelineTest(unittest.TestCase):
    def test_clean_text_removes_noise(self):
        self.assertEqual(clean_text(" 老师讲得很好！！！ T_T "), "老师讲得很好！！！")

    def test_tokenize_keeps_meaningful_words(self):
        tokens = tokenize("实验环境配置太复杂")
        self.assertTrue(tokens)

    def test_topic_detection(self):
        topics = detect_topics("实验环境配置太复杂，经常运行报错")
        self.assertIn("实验实践", topics)

    def test_keyword_extraction(self):
        keywords = keywords_only("作业太多了，作业提交时间也紧", top_k=2)
        self.assertIn("作业", keywords)

    def test_similarity(self):
        score = cosine_similarity("实验环境配置复杂", "实验配置步骤太麻烦")
        self.assertGreater(score, 0)

    def test_analyze_review(self):
        result = analyze_review("老师讲课很清楚，课堂互动很多", use_llm=False)
        self.assertEqual(result["sentiment"], "positive")
        self.assertIn("授课方式", result["topics"])

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


if __name__ == "__main__":
    unittest.main()
