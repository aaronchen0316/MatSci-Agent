from __future__ import annotations

from types import SimpleNamespace

import multiagent.sdk as harness_sdk
from multiagent.settings import MultiAgentSettings


def test_configure_sdk_uses_one_shared_client_and_disables_tracing(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    fake_sdk = SimpleNamespace(
        set_tracing_disabled=lambda **kwargs: calls.setdefault("tracing", kwargs),
        set_default_openai_client=lambda client, **kwargs: calls.setdefault("client", (client, kwargs)),
    )
    monkeypatch.setattr(harness_sdk, "load_openai_agents_sdk", lambda: fake_sdk)
    monkeypatch.setattr(harness_sdk, "AsyncOpenAI", lambda **kwargs: {"client": kwargs})
    settings = MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, api_key="token", base_url="https://example.test/v1")

    configured = harness_sdk.configure_sdk(settings)

    assert configured is fake_sdk
    assert calls["tracing"] == {"disabled": True}
    assert calls["client"] == (
        {"client": {"api_key": "token", "base_url": "https://example.test/v1"}},
        {"use_for_tracing": False},
    )
