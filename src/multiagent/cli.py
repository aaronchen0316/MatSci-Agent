from __future__ import annotations

import json

import typer
from rich.console import Console

from multiagent.model_preflight import prepare_live_models
from multiagent.settings import MultiAgentSettings
from multiagent.validation_repair import (
    blocked_validation_repair,
    run_live_validation,
    run_validation_repair,
    validation_repair_prerequisite_error,
)

app = typer.Typer(help="Validate and repair MatSci-Agent retrieval quality.")
console = Console()


@app.command("validate")
def validate() -> None:
    """Run the fixed eight-scenario live Materials Project validation."""

    settings = MultiAgentSettings.from_env()
    live_settings, preflight = prepare_live_models(settings)
    console.print_json(preflight.model_dump_json())
    if live_settings is None:
        raise typer.Exit(code=1)
    result = run_live_validation(live_settings)
    console.print_json(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=1)


@app.command("validate-repair")
def validate_repair(
    adopt: list[str] | None = typer.Option(None, "--adopt", help="Explicit retained fix/<issue> branch to review, retry, and publish."),
) -> None:
    """Validate, repair each live failure once, merge proven repairs, then validate again."""

    settings = MultiAgentSettings.from_env()
    error = validation_repair_prerequisite_error(settings)
    if error:
        console.print_json(json.dumps({"status": "blocked", "summary": error}))
        raise typer.Exit(code=1)
    live_settings, preflight = prepare_live_models(settings)
    if live_settings is None:
        report = blocked_validation_repair(settings, preflight, summary=preflight.summary)
    else:
        report = run_validation_repair(live_settings, preflight, adopt_branches=adopt)
    console.print_json(report.model_dump_json())
    if report.status != "pass":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
