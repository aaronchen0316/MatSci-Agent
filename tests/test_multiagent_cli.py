from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import multiagent.cli as multiagent_cli
from multiagent.schemas import (
    LiveEvalSuiteReport,
    ModelPreflightReport,
    RepairAuditReport,
    RepairSuiteReport,
)
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


def test_plan_command_renders_safe_runtime_config(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))

    result = runner.invoke(multiagent_cli.app, ["plan", "evaluate retrieval"])

    assert result.exit_code == 0
    assert '"objective": "evaluate retrieval"' in result.stdout
    assert '"max_agent_turns": 20' in result.stdout
    assert "artifact_root" in result.stdout


def test_eval_live_preflights_then_prints_suite_report(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda value: _preflight(value))
    monkeypatch.setattr(multiagent_cli, "_run_from_target_base", lambda value, action: action(value))
    monkeypatch.setattr(
        multiagent_cli,
        "run_live_suite",
        lambda value: LiveEvalSuiteReport(status="pass", artifact_dir=str(value.resolved_artifact_root)),
    )

    result = runner.invoke(multiagent_cli.app, ["eval-live"])

    assert result.exit_code == 0
    assert result.stdout.count('"status": "pass"') == 2


def test_eval_live_stops_after_blocked_model_preflight(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    blocked = ModelPreflightReport(
        status="blocked",
        primary_model="gpt-5.4-mini",
        attempts=["gpt-5.4-mini"],
        summary="proxy unavailable",
    )
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda _value: (None, blocked))

    result = runner.invoke(multiagent_cli.app, ["eval-live"])

    assert result.exit_code == 1
    assert "proxy unavailable" in result.stdout


def test_audit_repairs_persists_read_only_report(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    report = RepairAuditReport(status="fail", target_base_branch="main", target_base_sha="base")
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "audit_repair_branches", lambda *_args, **_kwargs: report)

    result = runner.invoke(multiagent_cli.app, ["audit-repairs"])

    assert result.exit_code == 1
    stored = list(settings.resolved_artifact_root.glob("*/repair_audit_report.json"))
    assert len(stored) == 1
    assert '"target_base_sha": "base"' in result.stdout


def test_repair_suite_blocks_before_model_call_when_prerequisites_fail(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "_repair_prerequisite_error", lambda _settings: "working tree dirty")

    result = runner.invoke(multiagent_cli.app, ["repair-suite"])

    assert result.exit_code == 1
    assert "working tree dirty" in result.stdout


def test_repair_suite_prints_blocked_preflight_report(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    blocked = ModelPreflightReport(
        status="blocked",
        primary_model="gpt-5.4-mini",
        attempts=["gpt-5.4-mini"],
        summary="model endpoint rejected request",
    )
    suite = RepairSuiteReport(
        status="blocked",
        summary=blocked.summary,
        model_preflight=blocked,
        audit_report=RepairAuditReport(status="pass", target_base_branch="main"),
    )
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "_repair_prerequisite_error", lambda _settings: None)
    monkeypatch.setattr(multiagent_cli, "prepare_live_models", lambda _value: (None, blocked))
    monkeypatch.setattr(multiagent_cli, "blocked_repair_suite", lambda *_args, **_kwargs: suite)

    result = runner.invoke(multiagent_cli.app, ["repair-suite"])

    assert result.exit_code == 1
    assert "model endpoint rejected request" in result.stdout
