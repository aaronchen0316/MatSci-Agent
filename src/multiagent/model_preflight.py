from __future__ import annotations

from dataclasses import replace
from typing import Callable

from openai import OpenAI

from multiagent.schemas import ModelPreflightReport
from multiagent.settings import MultiAgentSettings

PRIMARY_MODEL = "gpt-5.5"


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
        return None, ModelPreflightReport(
            status="blocked",
            primary_model=PRIMARY_MODEL,
            attempts=[PRIMARY_MODEL],
            summary=f"model preflight failed: {type(primary_error).__name__}",
        )
    return primary, ModelPreflightReport(
        status="pass",
        primary_model=PRIMARY_MODEL,
        selected_model=PRIMARY_MODEL,
        selected_product_model=PRIMARY_MODEL,
        attempts=[PRIMARY_MODEL],
        summary="primary model preflight passed",
    )
