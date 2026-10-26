from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from matsci_agent.cli import doctor, examples, render, transport
from matsci_agent.schemas import DiscoveryConstraints, DiscoveryRequest

app = typer.Typer(help="MatSci operator/demo CLI.")
example_app = typer.Typer(help="Built-in example requests.")
app.add_typer(example_app, name="examples")

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


@example_app.command("list")
def list_examples() -> None:
    for example in examples.EXAMPLES.values():
        console.print(f"  {example.name}: {example.description}")
        if example.prereq_note:
            console.print(f"  prereqs: {example.prereq_note}")


@example_app.command("run")
def run_example(
    name: str = typer.Argument(..., help="Built-in example name."),
    api_url: str | None = typer.Option(None, "--api-url"),
) -> None:
    example = examples.get_example(name)
    if example is None:
        _fail(f"Unknown example: {name}")

    notes: list[str] = []
    if api_url and example.enable_policy_filter:
        notes.append("HTTP mode uses server-side policy filtering. Remote LLM credentials are required for band-gap screening.")
    if example.name == "matgl_recalc":
        checks = doctor.run_doctor_checks(api_url=None)
        missing = [check.detail for check in checks if check.name in {"MatGL", "PyTorch", "PyG"} and check.status != "PASS"]
        if missing:
            notes.append("MatGL example conditional. Local prereqs incomplete.")
            notes.extend(missing)
    try:
        response = transport.run_summary(
            example.request,
            api_url=api_url,
            enable_policy_filter=example.enable_policy_filter,
        )
    except transport.CLITransportError as exc:
        _fail(str(exc))
    render.render_demo(console, example.request, response, notes=notes)


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
        max_energy_above_hull=max_energy_above_hull,
        top_k=top_k if top_k is not None else DiscoveryConstraints().top_k,
    )
    return DiscoveryRequest(research_goal=goal, constraints=constraints)


def _fail(message: str) -> None:
    console.print(Panel.fit(message, title="CLI Error", border_style="red"))
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """MatSci CLI."""
