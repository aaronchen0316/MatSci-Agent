from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from multiagent.live_suite import run_live_suite
from multiagent.orchestrator import MultiAgentHarness
from multiagent.publisher import publish_and_merge_repair
from multiagent.settings import MultiAgentSettings
from multiagent.tools import cleanup_worktree, create_target_base_worktree

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
                    "tool_root": str(settings.resolved_tool_root),
                    "target_repo": str(settings.resolved_target_repo),
                    "target_base_branch": settings.target_base_branch,
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
    result = _run_from_target_base(settings, lambda scoped: asyncio.run(MultiAgentHarness.build(scoped).run(objective)))
    console.print_json(result.model_dump_json())


@app.command("eval-live")
def eval_live() -> None:
    """Run credentialed live Materials Project regression scenarios."""

    settings = MultiAgentSettings.from_env()
    result = _run_from_target_base(settings, run_live_suite)
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
        cwd=str(settings.resolved_tool_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        return "tooling checkout must be clean before live repair"
    ref = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{settings.target_base_branch}"],
        cwd=str(settings.resolved_target_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return None if ref.returncode == 0 else f"base branch does not exist: {settings.target_base_branch}"


def _run_from_target_base(settings: MultiAgentSettings, action):
    created = create_target_base_worktree(settings)
    if created["status"] != "created":
        raise RuntimeError(f"unable to create target base worktree: {created.get('reason', 'unknown error')}")
    worktree_path = created["worktree_path"]
    try:
        return action(replace(settings, active_target_root=Path(worktree_path)))
    finally:
        cleanup_worktree(settings, worktree_path)


@app.command("repair-live")
def repair_live(scenario: str = typer.Option(..., "--scenario", help="One named live regression scenario.")) -> None:
    """Repair one credentialed live scenario in an isolated worktree."""

    from multiagent.live_suite import get_live_scenario

    settings = MultiAgentSettings.from_env()
    try:
        selected = get_live_scenario(scenario)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--scenario") from exc
    error = _repair_prerequisite_error(settings)
    if error:
        console.print_json(json.dumps({"status": "blocked", "scenario": selected.name, "summary": error}))
        raise typer.Exit(code=1)
    result = _run_from_target_base(
        settings,
        lambda scoped: asyncio.run(MultiAgentHarness.build(scoped).run(selected.query, scenario=selected)),
    )
    console.print_json(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=1)
    if result.branch_name is None or result.artifact_dir is None:
        return
    publication = publish_and_merge_repair(
        settings,
        branch_name=result.branch_name,
        artifact_dir=Path(result.artifact_dir),
    )
    console.print_json(publication.model_dump_json())
    if publication.status != "merged":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
