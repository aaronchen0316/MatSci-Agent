from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import multiagent.orchestrator as orchestrator
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


def test_runtime_builds_only_specialist_agents(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(orchestrator, "_agent_prompt", lambda name, _settings: f"prompt:{name}")
    sdk = FakeSDK()

    registry = orchestrator._build_agent_registry(
        sdk,
        MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path),
        ToolGroups(tester=[], critic=[], debugger=[], verifier=[]),
    )

    assert not hasattr(registry, "controller")
    assert [agent["name"] for agent in sdk.agents] == [
        "Retrieval Tester Agent",
        "Materials Query Critic Agent",
        "Codex Debugger Agent",
        "Final Verifier Agent",
    ]
    assert all(agent["output_type"]["strict_json_schema"] is False for agent in sdk.agents)


def test_runtime_configures_one_shared_client_and_disables_tracing(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    fake_sdk = SimpleNamespace(
        set_tracing_disabled=lambda **kwargs: calls.setdefault("tracing", kwargs),
        set_default_openai_client=lambda client, **kwargs: calls.setdefault("client", (client, kwargs)),
    )
    monkeypatch.setattr(orchestrator.importlib, "import_module", lambda _name: fake_sdk)
    monkeypatch.setattr(orchestrator, "AsyncOpenAI", lambda **kwargs: {"client": kwargs})
    settings = MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, api_key="token", base_url="https://example.test/v1")

    configured = orchestrator._configure_sdk(settings)

    assert configured is fake_sdk
    assert calls["tracing"] == {"disabled": True}
    assert calls["client"] == (
        {"client": {"api_key": "token", "base_url": "https://example.test/v1"}},
        {"use_for_tracing": False},
    )
