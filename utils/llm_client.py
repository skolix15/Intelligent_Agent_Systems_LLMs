from typing import Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception


def _is_retryable(exc: Exception) -> bool:
    """Only retry transient errors (rate limit, timeout, connection). Never retry 401/403."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (401, 403):
        return False
    msg = str(exc).lower()
    if "permission" in msg or "authentication" in msg or "invalid_api_key" in msg:
        return False
    return True


llm_retry = retry(
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


def get_llm(model: str):
    """Return the appropriate LangChain chat model based on the model name."""
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0)


def get_tokens_used(response: Any) -> int:
    """Extract total token count from a LangChain response, safely."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0
    if isinstance(meta, dict):
        return meta.get("total_tokens", 0)
    return getattr(meta, "total_tokens", 0)
