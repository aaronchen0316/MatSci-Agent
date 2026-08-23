from __future__ import annotations

from dataclasses import replace
from typing import Callable

from openai import OpenAI

from multiagent.schemas import ModelPreflightReport
from multiagent.settings import MultiAgentSettings

PRIMARY_MODEL = "gpt-5.4-mini"
FALLBACK_MODEL = "gpt-5.5"


def _is_unavailable_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "model" in text and any(
        marker in text
        for marker in ("unavailable", "not available", "not found", "does not exist", "model_not_found")
    )


def _probe_model(settings: MultiAgentSettings) -> None:
    if not settings.api_key:
        raise RuntimeError("missing OpenAI-compatible API key")
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=30.0)
    try:
        client.chat.completions.create(
            model=settings.model,
            temperature=0,
            max_tokens=1,
            messages=[{"role": "user", "content": "ok"}],
        )
    finally:
        client.close()


def prepare_live_models(
    settings: MultiAgentSettings,
    *,
    probe: Callable[[MultiAgentSettings], None] = _probe_model,
) -> tuple[MultiAgentSettings | None, ModelPreflightReport]:
    """Resolve live harness and product models through one proxy probe."""

    primary = replace(settings, model=PRIMARY_MODEL, product_model=PRIMARY_MODEL)
    try:
        probe(primary)
    except Exception as primary_error:
        if not _is_unavailable_model_error(primary_error):
            return None, ModelPreflightReport(
                status="blocked",
                primary_model=PRIMARY_MODEL,
                attempts=[PRIMARY_MODEL],
                summary=f"model preflight failed: {type(primary_error).__name__}",
            )
        fallback = replace(settings, model=FALLBACK_MODEL, product_model=FALLBACK_MODEL)
        try:
            probe(fallback)
        except Exception as fallback_error:
            return None, ModelPreflightReport(
                status="blocked",
                primary_model=PRIMARY_MODEL,
                attempts=[PRIMARY_MODEL, FALLBACK_MODEL],
                summary=f"fallback model preflight failed: {type(fallback_error).__name__}",
            )
        return fallback, ModelPreflightReport(
            status="fallback",
            primary_model=PRIMARY_MODEL,
            selected_model=FALLBACK_MODEL,
            selected_product_model=FALLBACK_MODEL,
            attempts=[PRIMARY_MODEL, FALLBACK_MODEL],
            summary="primary model explicitly unavailable; using fallback model",
        )
    return primary, ModelPreflightReport(
        status="pass",
        primary_model=PRIMARY_MODEL,
        selected_model=PRIMARY_MODEL,
        selected_product_model=PRIMARY_MODEL,
        attempts=[PRIMARY_MODEL],
        summary="primary model preflight passed",
    )
