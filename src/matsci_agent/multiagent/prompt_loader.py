from __future__ import annotations

from pathlib import Path


def load_agent_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "agent_specs" / f"{name}.md"
    return prompt_path.read_text()
