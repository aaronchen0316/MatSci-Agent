from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import multiagent.cli as multiagent_cli
from multiagent.schemas import LiveEvalSuiteReport, ModelPreflightReport, ValidationRepairReport
from multiagent.settings import MultiAgentSettings


runner = CliRunner()


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(
        tool_root=tmp_path,
        target_repo=tmp_path,
        artifact_root=tmp_path / "artifacts" / "multiagent-runs",
    )


def _preflight(settings: MultiAgentSettings) -> tuple[MultiAgentSettings, ModelPreflightReport]:
    return settings, ModelPreflightReport(
        status="pass",
        primary_model="gpt-5.4-mini",
        selected_model="gpt-5.4-mini",
        selected_product_model="gpt-5.4-mini",
        attempts=["gpt-5.4-mini"],
        summary="ok",
    )


def test_validate_preflights_then_prints_live_suite(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda value: _preflight(value))
    monkeypatch.setattr(
        multiagent_cli,
        "run_live_validation",
        lambda value: LiveEvalSuiteReport(status="pass", artifact_dir=str(value.resolved_artifact_root)),
    )

    result = runner.invoke(multiagent_cli.app, ["validate"])

    assert result.exit_code == 0
    assert result.stdout.count('"status": "pass"') == 2


def test_validate_stops_after_blocked_model_preflight(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    blocked = ModelPreflightReport(
        status="blocked",
        primary_model="gpt-5.4-mini",
        attempts=["gpt-5.4-mini"],
        summary="proxy unavailable",
    )
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda _value: (None, blocked))

    result = runner.invoke(multiagent_cli.app, ["validate"])

    assert result.exit_code == 1
    assert "proxy unavailable" in result.stdout


def test_validate_repair_blocks_before_model_call_when_prerequisites_fail(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "validation_repair_prerequisite_error", lambda _settings: "working tree dirty")

    result = runner.invoke(multiagent_cli.app, ["validate-repair"])

    assert result.exit_code == 1
    assert "working tree dirty" in result.stdout


def test_validate_repair_persists_blocked_preflight_report(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    blocked = ModelPreflightReport(
        status="blocked",
        primary_model="gpt-5.4-mini",
        attempts=["gpt-5.4-mini"],
        summary="model endpoint rejected request",
    )
    report = ValidationRepairReport(status="blocked", summary=blocked.summary, model_preflight=blocked)
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "validation_repair_prerequisite_error", lambda _settings: None)
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda _value: (None, blocked))
    monkeypatch.setattr(multiagent_cli, "blocked_validation_repair", lambda *_args, **_kwargs: report)

    result = runner.invoke(multiagent_cli.app, ["validate-repair"])

    assert result.exit_code == 1
    assert "model endpoint rejected request" in result.stdout


def test_validate_repair_forwards_explicit_adoption_branches(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    report = ValidationRepairReport(status="pass", summary="ok", model_preflight=_preflight(settings)[1])
    observed: dict[str, object] = {}
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "validation_repair_prerequisite_error", lambda _settings: None)
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda value: _preflight(value))

    def run(value, preflight, *, adopt_branches):
        observed["settings"] = value
        observed["preflight"] = preflight
        observed["adopt_branches"] = adopt_branches
        return report

    monkeypatch.setattr(multiagent_cli, "run_validation_repair", run)

    result = runner.invoke(
        multiagent_cli.app,
        ["validate-repair", "--adopt", "fix/formation", "--adopt", "fix/volume"],
    )

    assert result.exit_code == 0
    assert observed["adopt_branches"] == ["fix/formation", "fix/volume"]
