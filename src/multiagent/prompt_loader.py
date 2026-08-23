from __future__ import annotations

from pathlib import Path

from multiagent.settings import MultiAgentSettings


def load_agent_prompt(name: str, settings: MultiAgentSettings) -> str:
    prompt_path = settings.resolved_tool_root / "agent_specs" / f"{name}.md"
    return prompt_path.read_text()
