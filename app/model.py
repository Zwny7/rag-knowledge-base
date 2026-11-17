import json
import logging
from functools import lru_cache

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from . import settings

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def client() -> OpenAI:
    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL or None,
        timeout=12.0,
        max_retries=1,
    )


def complete(messages: list[dict], **options):
    if not settings.MODEL_READY:
        raise ModelError("请先在项目 .env 配置模型并重启项目。")
    try:
        response = client().chat.completions.create(
            model=settings.LLM_MODEL, messages=messages, max_tokens=900, **options
        )
        choice = response.choices[0]
        if choice.finish_reason in {"length", "content_filter"}:
            raise ModelError("模型没有返回完整结果，请缩短输入或改用本地模式。")
        return choice.message
    except (OpenAIError, IndexError) as exc:
        logger.warning("model_call_failed error_type=%s", type(exc).__name__)
        raise ModelError("模型服务调用失败，请检查配置或改用本地模式。") from exc


def parse_output(content: str | None, schema: type[BaseModel]):
    raw = (content or "").strip()
    if raw.startswith("```") and raw.endswith("```") and "\n" in raw:
        raw = raw.split("\n", 1)[1][:-3].strip()
    try:
        return schema.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise ModelError("模型返回的格式不符合要求，请重试或改用本地模式。") from exc
