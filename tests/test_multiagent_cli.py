from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import matsci_agent.multiagent.cli as multiagent_cli
from matsci_agent.multiagent.schemas import LiveEvalSuiteReport
from matsci_agent.multiagent.settings import MultiAgentSettings


runner = CliRunner()


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(repo_root=tmp_path, artifact_root=tmp_path / "artifacts" / "multiagent-runs")


def test_plan_command_renders_safe_runtime_config(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))

    result = runner.invoke(multiagent_cli.app, ["plan", "evaluate retrieval"])

    assert result.exit_code == 0
    assert '"objective": "evaluate retrieval"' in result.stdout
    assert "artifact_root" in result.stdout
    assert "enable_prs" not in result.stdout


def test_eval_live_command_prints_suite_report(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))
    monkeypatch.setattr(
        multiagent_cli,
        "run_live_suite",
        lambda settings: LiveEvalSuiteReport(status="pass", artifact_dir=str(settings.resolved_artifact_root)),
    )

    result = runner.invoke(multiagent_cli.app, ["eval-live"])

    assert result.exit_code == 0
    assert '"status": "pass"' in result.stdout


def test_eval_live_command_fails_when_live_evidence_is_blocked(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))
    monkeypatch.setattr(
        multiagent_cli,
        "run_live_suite",
        lambda settings: LiveEvalSuiteReport(status="blocked", artifact_dir=str(settings.resolved_artifact_root)),
    )

    result = runner.invoke(multiagent_cli.app, ["eval-live"])

    assert result.exit_code == 1
