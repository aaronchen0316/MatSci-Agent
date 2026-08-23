from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import multiagent.cli as multiagent_cli
from multiagent.schemas import HarnessRunReport, LiveEvalSuiteReport, PullRequestPublication
from multiagent.settings import MultiAgentSettings


runner = CliRunner()


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, artifact_root=tmp_path / "artifacts" / "multiagent-runs")


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
    monkeypatch.setattr(multiagent_cli, "_run_from_target_base", lambda settings, action: action(settings))
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
    monkeypatch.setattr(multiagent_cli, "_run_from_target_base", lambda settings, action: action(settings))
    monkeypatch.setattr(
        multiagent_cli,
        "run_live_suite",
        lambda settings: LiveEvalSuiteReport(status="blocked", artifact_dir=str(settings.resolved_artifact_root)),
    )

    result = runner.invoke(multiagent_cli.app, ["eval-live"])

    assert result.exit_code == 1


def test_run_command_prints_dual_review_result(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))
    monkeypatch.setattr(multiagent_cli, "_run_from_target_base", lambda settings, action: action(settings))

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


def test_repair_live_requires_target_base_to_match_origin(monkeypatch, tmp_path: Path):
    settings = MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, enable_live_mp=True, enable_git_write=True)

    def fake_run(args, **_kwargs):
        if args[:2] == ["git", "status"]:
            output = ""
        elif args[:2] == ["git", "show-ref"]:
            output = ""
        else:
            output = "local-sha\n" if args[-1] == "main" else "remote-sha\n"
        return __import__("subprocess").CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(multiagent_cli.subprocess, "run", fake_run)

    assert multiagent_cli._repair_prerequisite_error(settings) == "target base branch must match origin/main before live repair"


def test_repair_live_resolves_named_scenario(monkeypatch, tmp_path: Path):
    settings = MultiAgentSettings(
        tool_root=tmp_path, target_repo=tmp_path,
        artifact_root=tmp_path / "artifacts",
        enable_live_mp=True,
        enable_git_write=True,
        target_base_branch="multi-agent",
    )
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "_repair_prerequisite_error", lambda _settings: None)
    monkeypatch.setattr(multiagent_cli, "_run_from_target_base", lambda scoped, action: action(scoped))
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


def test_repair_live_auto_publishes_only_successful_repair(monkeypatch, tmp_path: Path):
    settings = MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, enable_live_mp=True, enable_git_write=True)
    monkeypatch.setattr(multiagent_cli.MultiAgentSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(multiagent_cli, "_repair_prerequisite_error", lambda _settings: None)
    monkeypatch.setattr(multiagent_cli, "_run_from_target_base", lambda scoped, action: action(scoped))
    observed = {}

    class FakeHarness:
        async def run(self, objective: str, *, scenario) -> HarnessRunReport:
            return HarnessRunReport(
                status="pass",
                summary=objective,
                next_step="merged automatically",
                stop_reason="dual_review_pass",
                attempt_count=2,
                branch_name="fix/volume",
                artifact_dir=str(tmp_path / "artifacts"),
            )

    def fake_publish(settings, **kwargs):
        observed["settings"] = settings
        observed.update(kwargs)
        return PullRequestPublication(
            status="merged",
            branch_name=kwargs["branch_name"],
            base_branch="main",
            summary="merged",
        )

    monkeypatch.setattr(multiagent_cli.MultiAgentHarness, "build", classmethod(lambda cls, _settings: FakeHarness()))
    monkeypatch.setattr(multiagent_cli, "publish_and_merge_repair", fake_publish)

    result = runner.invoke(multiagent_cli.app, ["repair-live", "--scenario", "volume"])

    assert result.exit_code == 0
    assert observed["branch_name"] == "fix/volume"
    assert observed["artifact_dir"] == tmp_path / "artifacts"
