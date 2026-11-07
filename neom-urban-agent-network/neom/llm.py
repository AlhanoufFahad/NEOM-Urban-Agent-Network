"""Groq-hosted LLM proposals via LangChain / ChatGroq, traced by LangSmith."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field


class ChargingProposal(BaseModel):
    count: int = Field(ge=0, description="Number of EVs to delay")
    steps: int = Field(ge=1, description="Number of simulation steps to delay")
    reasoning: str = Field(description="Brief explanation")


def configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


TRANSIENT_STATUS_CODES = {408, 500, 502, 503, 504}
# Avoid hammering the provider when the account has exhausted its quota.
_quota_blocked_until = 0.0


def _http_status(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"(?:HTTP\s*|status(?: code)?[=: ]+|\b)(400|401|403|404|408|413|422|429|500|502|503|504)\b", str(exc), re.I)
    return int(match.group(1)) if match else None


def _retryable(exc: Exception) -> bool:
    status = _http_status(exc)
    if status is not None:
        return status in TRANSIENT_STATUS_CODES
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name or "timed out" in str(exc).lower()


def _parse_proposal(content: Any) -> dict[str, Any]:
    """Parse untrusted model output; simulation later clamps proposals to safe bounds."""
    if not isinstance(content, str):
        raise ValueError("Model response was not text")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some providers add surrounding prose; accept only one JSON object.
        match = re.search(r"\{[^{}]*\}", cleaned, re.S)
        if not match:
            raise ValueError("Model did not return JSON") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return ChargingProposal.model_validate(parsed).model_dump()


def propose_charging_delay(context: dict[str, Any], model_name: str = "openai/gpt-oss-20b") -> dict[str, Any]:
    """Ask Groq LLM for a charging proposal, with bounded transient retries.

    429 is not retried. The Coordinator still validates count/steps before use.
    """
    global _quota_blocked_until
    if not configured():
        raise RuntimeError("GROQ_API_KEY is not set")
    if time.monotonic() < _quota_blocked_until:
        raise RuntimeError("HTTP 429: Groq cooldown active; using safe heuristic")

    from langchain_groq import ChatGroq

    llm = ChatGroq(model=model_name, timeout=45, max_retries=0)
    messages = [
        ("system", "You are an Energy Agent in a *simulated* smart city. Return ONLY one valid JSON "
         "object with integer keys count, steps and a short string reasoning. "
         "Keep count between 1 and the requested EV relief; keep steps between 1 and "
         "the maximum delay. Never add commentary or Markdown."),
        ("human", f"district={context['district']}; queue={context['queue']}; "
         f"requested_EV_relief={context['requested_count']}; grid_load_percent={context['grid_load']}; "
         f"grid_peak_percent={context['peak_threshold']}; congestion={context['congestion']}; "
         f"congestion_limit={context['congestion_limit']}; "
         f"max_delay_steps={context['max_delay_steps']}. Output a safe charging delay as JSON."),
    ]
    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            return _parse_proposal(response.content)
        except Exception as exc:
            if _http_status(exc) == 429:
                _quota_blocked_until = time.monotonic() + 60
                raise
            if attempt == 2 or not _retryable(exc):
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def explain_error(exc: Exception, model_name: str) -> str:
    """Safe, credential-free messages for the public dashboard."""
    code = _http_status(exc)
    kind = type(exc).__name__
    low = (kind + " " + str(exc)).lower()
    if code == 429:
        reason = "Groq rate/quota limit reached (429); pause requests or check Groq console; 60-second cooldown"
    elif code in (401, 403):
        reason = "Groq API key or permissions rejected; verify GROQ_API_KEY"
    elif code == 404 or "modelnotfound" in low:
        reason = f"model {model_name} is not available for your Groq account"
    elif code in (400, 413, 422):
        reason = "Groq rejected request or model parameters; check model and prompt"
    elif code in TRANSIENT_STATUS_CODES:
        reason = "Groq service error; retries exhausted; safe heuristic used"
    elif "timeout" in low or "connection" in low:
        reason = "Groq connection or timeout error; safe heuristic used"
    elif "json" in low or "validation" in low:
        reason = "LLM response invalid; safe heuristic used"
    elif "groq_api_key is not set" in low:
        reason = "GROQ_API_KEY missing"
    else:
        reason = "provider error; see local terminal for diagnosis"
    return f"{kind}{f' (HTTP {code})' if code else ''}: {reason}"
