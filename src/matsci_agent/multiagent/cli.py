from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel

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
                    "disable_tracing": settings.disable_tracing,
                    "enable_live_mp": settings.enable_live_mp,
                    "enable_git_write": settings.enable_git_write,
                    "enable_prs": settings.enable_prs,
                    "base_branch": settings.base_branch,
                    "github_repo": settings.github_repo,
                },
                indent=2,
            ),
            title="Multi-Agent Plan",
        )
    )


@app.command("run")
def run(objective: str) -> None:
    """Run controller agent.

    This command makes model calls. Keep env flags off until you are ready.
    """

    settings = MultiAgentSettings.from_env()
    harness = MultiAgentHarness.build(settings)
    result = asyncio.run(harness.run(objective))
    console.print_json(json.dumps(result))
