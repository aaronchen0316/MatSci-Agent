from __future__ import annotations

from pathlib import Path

import multiagent.factory as factory
from multiagent.factory import build_agent_registry
from multiagent.settings import MultiAgentSettings
from multiagent.tools import ToolGroups


class FakeSDK:
    def __init__(self) -> None:
        self.agents: list[dict[str, object]] = []

    def Agent(self, **kwargs):
        self.agents.append(kwargs)
        return kwargs

    @staticmethod
    def AgentOutputSchema(output_type, strict_json_schema: bool):
        return {"output_type": output_type, "strict_json_schema": strict_json_schema}


def test_factory_builds_only_specialist_agents(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(factory, "load_agent_prompt", lambda name, _settings: f"prompt:{name}")
    sdk = FakeSDK()
    registry = build_agent_registry(
        sdk,
        MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path),
        ToolGroups(shared=[], tester=[], critic=[], debugger=[], verifier=[]),
    )

    assert not hasattr(registry, "controller")
    assert [agent["name"] for agent in sdk.agents] == [
        "Retrieval Tester Agent",
        "Materials Query Critic Agent",
        "Codex Debugger Agent",
        "Final Verifier Agent",
    ]
    assert all(agent["output_type"]["strict_json_schema"] is False for agent in sdk.agents)
