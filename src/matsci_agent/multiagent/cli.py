from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from matsci_agent.multiagent.live_suite import run_live_suite
from matsci_agent.multiagent.orchestrator import MultiAgentHarness
from matsci_agent.multiagent.publisher import publish_pull_request
from matsci_agent.multiagent.settings import MultiAgentSettings

app = typer.Typer(help="Experimental multi-agent retrieval-repair harness.")
console = Console()


@app.command("plan")
def plan(objective: str) -> None:
    """Show effective config without making model calls."""

    settings = MultiAgentSettings.from_env()
    console.print(
        Panel.fit(
            json.dumps(
                {
                    "objective": objective,
                    "model": settings.model,
                    "base_url": settings.base_url,
                    "max_agent_turns": settings.max_agent_turns,
                    "disable_tracing": settings.disable_tracing,
                    "enable_live_mp": settings.enable_live_mp,
                    "enable_git_write": settings.enable_git_write,
                    "base_branch": settings.base_branch,
                    "repair_branch_prefix": settings.repair_branch_prefix,
                    "artifact_root": str(settings.resolved_artifact_root),
                },
                indent=2,
            ),
            title="Multi-Agent Plan",
        )
    )


@app.command("run")
def run(objective: str) -> None:
    """Run retrieval-repair harness.

    This command makes model calls. Keep env flags off until you are ready.
    """

    settings = MultiAgentSettings.from_env()
    harness = MultiAgentHarness.build(settings)
    result = asyncio.run(harness.run(objective))
    console.print_json(result.model_dump_json())


@app.command("eval-live")
def eval_live() -> None:
    """Run credentialed live Materials Project regression scenarios."""

    settings = MultiAgentSettings.from_env()
    result = run_live_suite(settings)
    console.print_json(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=1)


def _repair_prerequisite_error(settings: MultiAgentSettings) -> str | None:
    if not settings.enable_live_mp:
        return "MULTIAGENT_ENABLE_LIVE_MP=1 is required"
    if not settings.enable_git_write:
        return "MULTIAGENT_ENABLE_GIT_WRITE=1 is required"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(settings.repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        return "current checkout must be clean before live repair"
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(settings.repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if branch.returncode != 0 or branch.stdout.strip() != settings.base_branch:
        return f"current checkout must be base branch {settings.base_branch}"
    ref = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{settings.base_branch}"],
        cwd=str(settings.repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return None if ref.returncode == 0 else f"base branch does not exist: {settings.base_branch}"


@app.command("repair-live")
def repair_live(scenario: str = typer.Option(..., "--scenario", help="One named live regression scenario.")) -> None:
    """Repair one credentialed live scenario in an isolated worktree."""

    from matsci_agent.multiagent.live_suite import get_live_scenario

    settings = MultiAgentSettings.from_env()
    try:
        selected = get_live_scenario(scenario)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--scenario") from exc
    error = _repair_prerequisite_error(settings)
    if error:
        console.print_json(json.dumps({"status": "blocked", "scenario": selected.name, "summary": error}))
        raise typer.Exit(code=1)
    harness = MultiAgentHarness.build(settings)
    result = asyncio.run(harness.run(selected.query, scenario=selected))
    console.print_json(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=1)


@app.command("publish-pr")
def publish_pr(
    branch_name: str,
    base: str | None = typer.Option(None, "--base", help="Pull-request base branch."),
    artifact_dir: Path | None = typer.Option(None, "--artifact-dir", exists=True, file_okay=False),
    validation_only: bool = typer.Option(False, "--validation-only"),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    """Push one guarded repair branch and create a draft GitHub pull request."""

    settings = MultiAgentSettings.from_env()
    result = publish_pull_request(
        settings,
        branch_name=branch_name,
        base_branch=base or settings.base_branch,
        artifact_dir=artifact_dir,
        validation_only=validation_only,
        reason=reason,
    )
    console.print_json(result.model_dump_json())
    if result.status != "published":
        raise typer.Exit(code=1)
