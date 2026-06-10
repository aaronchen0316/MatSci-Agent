from __future__ import annotations

import importlib
from types import ModuleType

from openai import AsyncOpenAI

from matsci_agent.multiagent.settings import MultiAgentSettings


def load_openai_agents_sdk() -> ModuleType:
    """Import `openai-agents` SDK module.

    Prompt specs live under `agent_specs/`, so import-shadowing workaround is
    no longer needed.
    """

    try:
        return importlib.import_module("agents")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenAI Agents SDK not installed. Run `uv sync --extra dev --extra agents`."
        ) from exc


def configure_sdk(settings: MultiAgentSettings) -> ModuleType:
    sdk = load_openai_agents_sdk()

    if settings.disable_tracing:
        sdk.set_tracing_disabled(disabled=True)

    if settings.api_key:
        client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        # One shared client is enough for controller + all sub-agents.
        sdk.set_default_openai_client(client, use_for_tracing=not settings.disable_tracing)

    return sdk
