from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from multiagent.artifacts import HarnessArtifactStore
from multiagent.audit import audit_repair_branches
from multiagent.live_suite import run_live_suite
from multiagent.model_preflight import prepare_live_models
from multiagent.orchestrator import MultiAgentHarness
from multiagent.settings import MultiAgentSettings
from multiagent.suite import (
    blocked_repair_suite,
    repair_prerequisite_error,
    run_from_target_base,
    run_repair_suite,
    run_single_repair,
)

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
    live_settings, preflight = prepare_live_models(settings)
    console.print_json(preflight.model_dump_json())
    if live_settings is None:
        raise typer.Exit(code=1)
    result = _run_from_target_base(live_settings, run_live_suite)
    console.print_json(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=1)


def _repair_prerequisite_error(settings: MultiAgentSettings) -> str | None:
    return repair_prerequisite_error(settings)


def _run_from_target_base(settings: MultiAgentSettings, action):
    return run_from_target_base(settings, action)


@app.command("audit-repairs")
def audit_repairs(branches: list[str] = typer.Argument(None, help="Optional local repair branches to audit.")) -> None:
    """Read-only readiness audit for retained and current repair branches."""

    settings = MultiAgentSettings.from_env()
    report = audit_repair_branches(settings, branch_names=branches or None)
    store = HarnessArtifactStore.create(settings, "multiagent_repair_audit")
    store.write_model("repair_audit_report.json", report)
    payload = report.model_copy(update={"target_base_sha": report.target_base_sha})
    console.print_json(payload.model_dump_json())
    if report.status != "pass":
        raise typer.Exit(code=1)


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
    live_settings, preflight = prepare_live_models(settings)
    console.print_json(preflight.model_dump_json())
    if live_settings is None:
        raise typer.Exit(code=1)
    result, publication = run_single_repair(live_settings, selected)
    console.print_json(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=1)
    if publication is not None:
        console.print_json(publication.model_dump_json())
    if publication is not None and publication.status != "merged":
        raise typer.Exit(code=1)


@app.command("repair-suite")
def repair_suite() -> None:
    """Audit retained branches, repair live failures, merge guarded product fixes, and retest all scenarios."""

    settings = MultiAgentSettings.from_env()
    error = _repair_prerequisite_error(settings)
    if error:
        console.print_json(json.dumps({"status": "blocked", "summary": error}))
        raise typer.Exit(code=1)
    live_settings, preflight = prepare_live_models(settings)
    if live_settings is None:
        report = blocked_repair_suite(settings, preflight, summary=preflight.summary)
    else:
        report = run_repair_suite(live_settings, preflight)
    console.print_json(report.model_dump_json())
    if report.status != "pass":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
