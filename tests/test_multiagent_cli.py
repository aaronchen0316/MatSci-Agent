from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import matsci_agent.multiagent.cli as multiagent_cli
from matsci_agent.multiagent.schemas import HarnessRunReport, LiveEvalSuiteReport, PullRequestPublication
from matsci_agent.multiagent.settings import MultiAgentSettings


runner = CliRunner()


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(repo_root=tmp_path, artifact_root=tmp_path / "artifacts" / "multiagent-runs")


def test_plan_command_renders_safe_runtime_config(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))

    result = runner.invoke(multiagent_cli.app, ["plan", "evaluate retrieval"])

    assert result.exit_code == 0
    assert '"objective": "evaluate retrieval"' in result.stdout
    assert '"max_agent_turns": 20' in result.stdout
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


def test_run_command_prints_dual_review_result(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))

    class FakeHarness:
        async def run(self, objective: str) -> HarnessRunReport:
            return HarnessRunReport(
                status="pass",
                summary=objective,
                next_step="review artifacts",
                stop_reason="dual_review_pass",
                attempt_count=1,
            )

    monkeypatch.setattr(multiagent_cli.MultiAgentHarness, "build", classmethod(lambda cls, settings: FakeHarness()))

    result = runner.invoke(multiagent_cli.app, ["run", "evaluate retrieval"])

    assert result.exit_code == 0
    assert '"stop_reason": "dual_review_pass"' in result.stdout


def test_repair_live_requires_live_and_git_mutation_flags(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))

    result = runner.invoke(multiagent_cli.app, ["repair-live", "--scenario", "volume"])

    assert result.exit_code == 1
    assert "MULTIAGENT_ENABLE_LIVE_MP=1 is required" in result.stdout


def test_repair_live_resolves_named_scenario(monkeypatch, tmp_path: Path):
    settings = MultiAgentSettings(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        enable_live_mp=True,
        enable_git_write=True,
        base_branch="multi-agent",
    )
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "_repair_prerequisite_error", lambda _settings: None)
    observed = {}

    class FakeHarness:
        async def run(self, objective: str, *, scenario) -> HarnessRunReport:
            observed["objective"] = objective
            observed["scenario"] = scenario
            return HarnessRunReport(
                status="pass",
                summary="ok",
                next_step="review",
                stop_reason="dual_review_pass",
                attempt_count=1,
            )

    monkeypatch.setattr(multiagent_cli.MultiAgentHarness, "build", classmethod(lambda cls, _settings: FakeHarness()))

    result = runner.invoke(multiagent_cli.app, ["repair-live", "--scenario", "volume"])

    assert result.exit_code == 0
    assert observed["scenario"].name == "volume"
    assert observed["objective"] == observed["scenario"].query


def test_publish_pr_forwards_validation_only_contract(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))
    observed = {}

    def fake_publish(settings, **kwargs):
        observed["settings"] = settings
        observed.update(kwargs)
        return PullRequestPublication(
            status="published",
            branch_name=kwargs["branch_name"],
            base_branch=kwargs["base_branch"],
            validation_only=True,
            summary="draft created",
        )

    monkeypatch.setattr(multiagent_cli, "publish_pull_request", fake_publish)

    result = runner.invoke(
        multiagent_cli.app,
        ["publish-pr", "retrieval-fix-retry", "--validation-only", "--reason", "synthetic fixture branch"],
    )

    assert result.exit_code == 0
    assert observed["validation_only"] is True
    assert observed["reason"] == "synthetic fixture branch"
