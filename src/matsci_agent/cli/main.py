from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from matsci_agent.cli import doctor, render, scenarios, transport
from matsci_agent.schemas import DiscoveryConstraints, DiscoveryRequest

app = typer.Typer(help="MatSci operator/demo CLI.")
scenario_app = typer.Typer(help="Built-in demo scenarios.")
app.add_typer(scenario_app, name="scenarios")

console = Console()


@app.command()
def demo(
    research_goal: str | None = typer.Argument(None, help="Natural-language materials query."),
    top_k: int | None = typer.Option(None, "--top-k"),
    min_band_gap_ev: float | None = typer.Option(None, "--min-band-gap-ev"),
    max_energy_above_hull: float | None = typer.Option(None, "--max-energy-above-hull"),
    banned_element: list[str] | None = typer.Option(None, "--banned-element"),
    required_element: list[str] | None = typer.Option(None, "--required-element"),
    calculate_matgl: bool = typer.Option(False, "--calculate-matgl"),
    api_url: str | None = typer.Option(None, "--api-url"),
) -> None:
    request = _build_request(
        research_goal,
        top_k=top_k,
        min_band_gap_ev=min_band_gap_ev,
        max_energy_above_hull=max_energy_above_hull,
        banned_element=banned_element,
        required_element=required_element,
        calculate_matgl=calculate_matgl,
    )
    try:
        response = transport.run_summary(request, api_url=api_url)
    except transport.CLITransportError as exc:
        _fail(str(exc))
    render.render_demo(console, request, response)


@app.command()
def operator(
    research_goal: str | None = typer.Argument(None, help="Natural-language materials query."),
    top_k: int | None = typer.Option(None, "--top-k"),
    min_band_gap_ev: float | None = typer.Option(None, "--min-band-gap-ev"),
    max_energy_above_hull: float | None = typer.Option(None, "--max-energy-above-hull"),
    banned_element: list[str] | None = typer.Option(None, "--banned-element"),
    required_element: list[str] | None = typer.Option(None, "--required-element"),
    calculate_matgl: bool = typer.Option(False, "--calculate-matgl"),
    api_url: str | None = typer.Option(None, "--api-url"),
) -> None:
    request = _build_request(
        research_goal,
        top_k=top_k,
        min_band_gap_ev=min_band_gap_ev,
        max_energy_above_hull=max_energy_above_hull,
        banned_element=banned_element,
        required_element=required_element,
        calculate_matgl=calculate_matgl,
    )
    try:
        response = transport.run_full(request, api_url=api_url)
    except transport.CLITransportError as exc:
        _fail(str(exc))
    render.render_operator(console, response)


@app.command("doctor")
def doctor_cmd(
    api_url: str | None = typer.Option(None, "--api-url"),
) -> None:
    checks = doctor.run_doctor_checks(api_url=api_url)
    render.render_doctor(console, checks)
    if doctor.has_failures(checks):
        raise typer.Exit(code=1)


@scenario_app.command("list")
def list_scenarios() -> None:
    for scenario in scenarios.SCENARIOS.values():
        console.print(f"  {scenario.name}: {scenario.description}")
        if scenario.prereq_note:
            console.print(f"  prereqs: {scenario.prereq_note}")


@scenario_app.command("run")
def run_scenario(
    name: str = typer.Argument(..., help="Built-in scenario name."),
    api_url: str | None = typer.Option(None, "--api-url"),
) -> None:
    scenario = scenarios.get_scenario(name)
    if scenario is None:
        _fail(f"Unknown scenario: {name}")

    notes: list[str] = []
    if api_url and scenario.enable_policy_filter:
        notes.append("HTTP mode uses server-side policy filtering. Remote LLM credentials are required for band-gap screening.")
    if scenario.name == "matgl_recalc":
        checks = doctor.run_doctor_checks(api_url=None)
        missing = [check.detail for check in checks if check.name in {"MatGL", "PyTorch", "DGL"} and check.status != "PASS"]
        if missing:
            notes.append("MatGL scenario conditional. Local prereqs incomplete.")
            notes.extend(missing)
    try:
        response = transport.run_summary(
            scenario.request,
            api_url=api_url,
            enable_policy_filter=scenario.enable_policy_filter,
        )
    except transport.CLITransportError as exc:
        _fail(str(exc))
    render.render_demo(console, scenario.request, response, notes=notes)


def _build_request(
    research_goal: str | None,
    *,
    top_k: int | None,
    min_band_gap_ev: float | None,
    max_energy_above_hull: float | None,
    banned_element: list[str] | None,
    required_element: list[str] | None,
    calculate_matgl: bool,
) -> DiscoveryRequest:
    goal = research_goal or typer.prompt("Research goal")
    constraints = DiscoveryConstraints(
        banned_elements=banned_element or [],
        required_elements=required_element or [],
        min_band_gap_ev=min_band_gap_ev,
        calculate_matgl=calculate_matgl,
        max_energy_above_hull=(
            max_energy_above_hull
            if max_energy_above_hull is not None
            else DiscoveryConstraints().max_energy_above_hull
        ),
        top_k=top_k if top_k is not None else DiscoveryConstraints().top_k,
    )
    return DiscoveryRequest(research_goal=goal, constraints=constraints)


def _fail(message: str) -> None:
    console.print(Panel.fit(message, title="CLI Error", border_style="red"))
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """MatSci CLI."""
