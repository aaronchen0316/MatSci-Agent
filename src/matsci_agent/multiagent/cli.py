from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel

from matsci_agent.multiagent.live_suite import run_live_suite
from matsci_agent.multiagent.orchestrator import MultiAgentHarness
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
