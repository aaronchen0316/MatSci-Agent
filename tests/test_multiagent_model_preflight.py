from __future__ import annotations

from multiagent.model_preflight import FALLBACK_MODEL, PRIMARY_MODEL, prepare_live_models
from multiagent.settings import MultiAgentSettings


def _settings(tmp_path):
    return MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, api_key="key", base_url="https://proxy.example/v1")


def test_model_preflight_uses_primary_for_harness_and_product(tmp_path):
    seen: list[str] = []

    settings, report = prepare_live_models(_settings(tmp_path), probe=lambda value: seen.append(value.model))

    assert seen == [PRIMARY_MODEL]
    assert settings is not None
    assert settings.model == PRIMARY_MODEL
    assert settings.product_model == PRIMARY_MODEL
    assert report.status == "pass"


def test_model_preflight_falls_back_only_for_explicit_unavailable_model(tmp_path):
    seen: list[str] = []

    def probe(settings):
        seen.append(settings.model)
        if settings.model == PRIMARY_MODEL:
            raise RuntimeError("model not found")

    settings, report = prepare_live_models(_settings(tmp_path), probe=probe)

    assert seen == [PRIMARY_MODEL, FALLBACK_MODEL]
    assert settings is not None
    assert settings.model == FALLBACK_MODEL
    assert settings.product_model == FALLBACK_MODEL
    assert report.status == "fallback"


def test_model_preflight_blocks_other_proxy_failures_without_fallback(tmp_path):
    seen: list[str] = []

    def probe(settings):
        seen.append(settings.model)
        raise RuntimeError("proxy connection refused")

    settings, report = prepare_live_models(_settings(tmp_path), probe=probe)

    assert settings is None
    assert seen == [PRIMARY_MODEL]
    assert report.status == "blocked"
