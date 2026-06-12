"""带确定性本地兜底的大模型 API 客户端。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 仅在依赖未安装时使用
    load_dotenv = None

try:
    import requests
except ImportError:  # pragma: no cover - 仅在依赖未安装时使用
    requests = None


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    model: str = "ecnu-plus"
    timeout: int = 20
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        if load_dotenv is not None:
            dotenv_path = Path.cwd() / ".env"
            if not dotenv_path.exists():
                dotenv_path = Path(__file__).resolve().parents[1] / ".env"
            load_dotenv(dotenv_path=dotenv_path)

        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://chat.ecnu.edu.cn/open/api/v1"),
            model=os.getenv("LLM_MODEL", "ecnu-plus"),
            timeout=int(os.getenv("LLM_TIMEOUT", "20")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            retries=int(os.getenv("LLM_RETRIES", "2")),
        )


REVIEW_ADVICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "description": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["aspect", "description", "evidence"],
            },
        },
        "suggestions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["aspect", "suggestion", "evidence"],
            },
        },
        "risk_level": {"type": "string", "enum": ["low", "middle", "high"]},
    },
    "required": ["summary", "problems", "suggestions", "risk_level"],
}

SYSTEM_PROMPT = """你是高校课程评价分析助手。请只根据用户提供的结构化 NLP 结果、课程维度证据和相似评论生成建议。
禁止编造学生没有提到的问题。输出必须是合法 JSON 对象，必须包含 summary、problems、suggestions、risk_level 四个字段。
problems 可以为空数组，但字段不能省略；suggestions 至少 1 条。
正面评价也必须给出维护型或推广型建议。aspect 必须优先来自 topic_evidence.aspect。
evidence 必须引用 review_text、topic_evidence.evidence 或 similar_reviews.text 中的原文片段。
表达要自然，避免直接堆叠关键词，避免重复课程维度。"""


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def build_single_review_prompt(analysis: dict[str, Any]) -> str:
    """为单条评价构建结构化提示词。"""

    topic_evidence = json.dumps(analysis.get("topic_evidence", []), ensure_ascii=False)
    similar_reviews = json.dumps(analysis.get("similar_reviews", []), ensure_ascii=False)
    topics = "、".join(_unique_strings(analysis.get("topics", [])))
    keywords = "、".join(_unique_strings(analysis.get("keywords", []))[:6])
    return f"""请根据下面的课程评价结构化结果，生成面向教师或课程管理者的自然中文反馈建议。

review_text：{analysis.get("text", "")}
语言类型：{analysis.get("language", "unknown")}
情感倾向：{analysis.get("sentiment", "")}
课程维度：{topics}
关键词：{keywords}
主题证据：{topic_evidence}
相似评论：{similar_reviews}

输出要求：
1. 只输出 JSON，不要 Markdown，不要解释。
2. JSON 包含 summary、problems、suggestions、risk_level，risk_level 只能是 low、middle、high。
3. problems 每项包含 aspect、description、evidence；suggestions 每项包含 aspect、suggestion、evidence。
4. aspect 优先使用“课程维度”中的名称。
5. summary 用 1 句自然中文说明整体反馈。
6. problems 和 suggestions 最多各 2 条，不要重复同一课程维度。
7. 不要把关键词列表直接拼成句子；如果评价是英文或中英混合，也用自然中文总结。"""


def local_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """在 API 不可用时生成稳定的本地总结。"""

    sentiment = analysis.get("sentiment", "neutral")
    language = analysis.get("language", "unknown")
    topics = _unique_strings(analysis.get("topics", []))
    topic_text = "、".join(topics[:3]) if topics else "课程体验"
    aspect = topics[0] if topics else "课程体验"
    evidence_items = analysis.get("topic_evidence", [])
    evidence = ""
    if evidence_items:
        evidence = str(evidence_items[0].get("evidence", ""))
    if not evidence:
        evidence = str(analysis.get("text", ""))[:80]

    if sentiment == "positive":
        summary = f"学生整体反馈较积极，主要认可{topic_text}方面。"
        risk_level = "low"
        problem_text = "暂未发现明确问题，可继续关注学生认可点是否能稳定保持。"
        suggestion_text = f"保持{topic_text}中的有效做法，并在后续课程中继续收集学生反馈。"
    elif sentiment == "negative":
        summary = f"学生反馈偏负面，问题集中在{topic_text}方面。"
        risk_level = "high"
        problem_text = f"需要优先核实{topic_text}中被学生明确指出的不便。"
        suggestion_text = f"针对{topic_text}制定具体调整措施，降低学生学习阻力。"
    else:
        summary = f"学生反馈较为中性，涉及{topic_text}，同时包含肯定和改进建议。"
        risk_level = "middle"
        problem_text = f"需要把{topic_text}中的认可点和不便之处分开处理。"
        suggestion_text = f"保留学生认可的做法，同时针对{topic_text}中提到的不便安排改进。"

    return {
        "summary": summary,
        "problems": [
            {
                "aspect": aspect,
                "description": problem_text,
                "evidence": evidence,
            }
        ],
        "suggestions": [
            {
                "aspect": aspect,
                "suggestion": suggestion_text,
                "evidence": evidence,
            }
        ],
        "risk_level": risk_level,
        "language": language,
        "source": "local_fallback",
    }


def _parse_json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(cleaned[start:])
        if not isinstance(result, dict):
            raise ValueError("LLM response JSON root must be an object")
        return result


def _validate_review_advice(result: dict[str, Any]) -> None:
    for key in REVIEW_ADVICE_SCHEMA["required"]:
        if key not in result:
            raise ValueError(f"LLM response missing required key: {key}")
    if result["risk_level"] not in {"low", "middle", "high"}:
        raise ValueError("LLM response has invalid risk_level")
    for key in ["problems", "suggestions"]:
        if not isinstance(result[key], list):
            raise ValueError(f"LLM response key must be a list: {key}")
        if key == "suggestions" and not result[key]:
            raise ValueError("LLM response suggestions must contain at least one item")
        for item in result[key]:
            if not isinstance(item, dict):
                raise ValueError(f"LLM response list item must be an object: {key}")
            required = ["aspect", "evidence"]
            if key == "problems":
                required.append("description")
            else:
                required.append("suggestion")
            for item_key in required:
                if not item.get(item_key):
                    raise ValueError(f"LLM response item missing required key: {item_key}")


def call_llm_json(prompt: str, config: LLMConfig | None = None) -> dict[str, Any]:
    """调用配置的大模型聊天接口，并解析 JSON 输出。"""

    config = config or LLMConfig.from_env()
    if not config.api_key or requests is None:
        raise RuntimeError("未配置大模型 API")

    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    last_error: Exception | None = None
    for _ in range(max(0, config.retries) + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=config.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("LLM response content is empty")
            result = _parse_json_content(content)
            _validate_review_advice(result)
            return result
        except Exception as exc:
            last_error = exc

    raise RuntimeError("LLM API call failed") from last_error


def generate_review_advice(analysis: dict[str, Any], config: LLMConfig | None = None) -> dict[str, Any]:
    """生成大模型建议，失败时回退到本地模板。"""

    prompt = build_single_review_prompt(analysis)
    try:
        result = call_llm_json(prompt, config=config)
        result.setdefault("source", "llm_api")
        return result
    except Exception as exc:
        fallback = local_summary(analysis)
        fallback["fallback_reason"] = exc.__class__.__name__
        return fallback
