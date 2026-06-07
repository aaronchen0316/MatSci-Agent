from __future__ import annotations

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from matsci_agent.cli.doctor import DoctorCheck
from matsci_agent.schemas import DiscoveryFullResponse, DiscoveryRequest, DiscoverySummaryResponse


def render_demo(
    console: Console,
    request: DiscoveryRequest,
    response: DiscoverySummaryResponse,
    notes: list[str] | None = None,
) -> None:
    console.print(
        Panel.fit(
            f"[bold]Status:[/bold] {response.status}\n[bold]Goal:[/bold] {request.research_goal}",
            title="MatSci Demo",
        )
    )
    _render_notes(console, notes)
    if response.unsupported_reason:
        console.print(Panel.fit(response.unsupported_reason, title="Unsupported"))
    if response.messages:
        console.print(Panel.fit("\n".join(response.messages), title="Messages"))

    table = Table(title="Candidates")
    table.add_column("Rank", justify="right")
    table.add_column("Material")
    table.add_column("Formula")
    if request.constraints.calculate_matgl:
        table.add_column("MP Gap", justify="right")
        table.add_column("MatGL Gap", justify="right")
    else:
        table.add_column("Band Gap (eV)", justify="right")
    table.add_column("Source")
    table.add_column("Entries", justify="right")
    table.add_column("Hull")
    table.add_column("Stable")
    table.add_column("Properties")
    for idx, candidate in enumerate(response.candidates, start=1):
        row = [
            str(idx),
            candidate.material_id,
            candidate.formula,
        ]
        if request.constraints.calculate_matgl:
            row.extend([
                _fmt(candidate.mp_band_gap_ev),
                _fmt(candidate.matgl_band_gap_ev),
            ])
        else:
            row.append(f"{candidate.band_gap_ev:.3f}")
        row.extend([
            candidate.band_gap_source or "-",
            str(candidate.entry_count),
            _fmt(candidate.energy_above_hull),
            _fmt(candidate.is_stable),
            _fmt_properties(candidate.properties),
        ])
        table.add_row(*row)
    if response.candidates:
        console.print(table)
    else:
        console.print(Panel.fit("No candidates returned.", title="Candidates"))


def render_operator(
    console: Console,
    response: DiscoveryFullResponse,
    notes: list[str] | None = None,
) -> None:
    console.print(
        Panel.fit(
            f"[bold]Status:[/bold] {response.status}\n[bold]Goal:[/bold] {response.research_goal}",
            title="MatSci Operator",
        )
    )
    _render_notes(console, notes)
    console.print(
        Panel.fit(
            JSON.from_data(response.constraints.model_dump(mode="json")),
            title="Effective Constraints",
        )
    )
    if response.discovery_plan is not None:
        console.print(
            Panel.fit(
                JSON.from_data(response.discovery_plan.model_dump(mode="json")),
                title="Discovery Plan",
            )
        )
    if response.capability_assessment is not None:
        console.print(
            Panel.fit(
                JSON.from_data(response.capability_assessment.model_dump(mode="json")),
                title="Capability Assessment",
            )
        )
    _render_candidate_table(console, "Raw Candidates", response.raw_candidates)
    _render_candidate_table(console, "Filtered Candidates", response.filtered_candidates)
    _render_search_space_targets(console, response.search_space_targets)
    _render_filter_records(console, response.filter_records)
    _render_ranked_candidates(console, response)
    console.print(
        Panel.fit(
            JSON.from_data([item.model_dump(mode="json") for item in response.provenance]),
            title="Provenance",
        )
    )
    console.print(
        Panel.fit(
            JSON.from_data(
                {
                    "messages": response.messages,
                    "report_summary": (
                        response.report_summary.model_dump(mode="json")
                        if response.report_summary is not None
                        else None
                    ),
                }
            ),
            title="Messages / Caveats",
        )
    )


def render_doctor(console: Console, checks: list[DoctorCheck]) -> None:
    table = Table(title="MatSci Doctor")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Detail")
    for check in checks:
        style = {
            "PASS": "green",
            "WARN": "yellow",
            "FAIL": "red",
        }.get(check.status, "white")
        table.add_row(f"[{style}]{check.status}[/{style}]", check.name, check.detail)
    console.print(table)


def _render_candidate_table(console: Console, title: str, candidates: list) -> None:
    table = Table(title=title)
    table.add_column("Material")
    table.add_column("Formula")
    table.add_column("Source")
    table.add_column("Entries", justify="right")
    table.add_column("MP Gap")
    table.add_column("Hull")
    table.add_column("Sites")
    table.add_column("Properties")
    for candidate in candidates:
        table.add_row(
            candidate.material_id,
            candidate.formula,
            candidate.source,
            str(candidate.entry_count),
            _fmt(candidate.features.get("mp_band_gap_ev")),
            _fmt(candidate.features.get("mp_energy_above_hull")),
            _fmt(candidate.features.get("nsites")),
            _fmt_properties(candidate.features.get("properties", {})),
        )
    if candidates:
        console.print(table)
    else:
        console.print(Panel.fit("None", title=title))


def _render_search_space_targets(console: Console, targets: list) -> None:
    table = Table(title="Search Space Targets")
    table.add_column("Formula")
    table.add_column("Chemsys")
    table.add_column("Confidence", justify="right")
    table.add_column("Rationale")
    for target in targets:
        table.add_row(
            target.normalized_formula,
            target.chemsys,
            f"{target.confidence:.2f}",
            target.rationale or "-",
        )
    if targets:
        console.print(table)
    else:
        console.print(Panel.fit("None", title="Search Space Targets"))


def _render_filter_records(console: Console, records: list) -> None:
    table = Table(title="Filter Records")
    table.add_column("Material")
    table.add_column("Formula")
    table.add_column("Passed")
    table.add_column("Policy")
    table.add_column("Reasons")
    for record in records:
        table.add_row(
            record.candidate.material_id,
            record.candidate.formula,
            _fmt(record.passed),
            record.policy,
            "; ".join(record.reasons) if record.reasons else "-",
        )
    if records:
        console.print(table)
    else:
        console.print(Panel.fit("None", title="Filter Records"))


def _render_ranked_candidates(console: Console, response: DiscoveryFullResponse) -> None:
    table = Table(title="Ranked Candidates")
    table.add_column("Rank", justify="right")
    table.add_column("Material")
    table.add_column("Formula")
    if _response_uses_matgl_columns(response):
        table.add_column("MP Gap", justify="right")
        table.add_column("MatGL Gap", justify="right")
    else:
        table.add_column("Band Gap (eV)", justify="right")
    table.add_column("Backend")
    table.add_column("Entries", justify="right")
    table.add_column("Stable")
    table.add_column("Score", justify="right")
    for candidate in response.candidates:
        row = [
            str(candidate.rank),
            candidate.candidate.material_id,
            candidate.candidate.formula,
        ]
        if _response_uses_matgl_columns(response):
            row.extend(
                [
                    _fmt(candidate.candidate.features.get("mp_band_gap_ev")),
                    _fmt(candidate.candidate.features.get("matgl_band_gap_ev")),
                ]
            )
        else:
            row.append(f"{candidate.predicted_properties.band_gap_ev:.3f}")
        row.extend(
            [
                candidate.predicted_properties.backend,
                str(candidate.candidate.entry_count),
                _fmt(candidate.stability.is_stable),
                f"{candidate.score:.3f}",
            ]
        )
        table.add_row(*row)
    if response.candidates:
        console.print(table)
    else:
        console.print(Panel.fit("None", title="Ranked Candidates"))


def _render_notes(console: Console, notes: list[str] | None) -> None:
    if notes:
        console.print(Panel.fit("\n".join(notes), title="Notes"))


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_properties(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    parts = []
    for key, item in value.items():
        if item is None:
            continue
        parts.append(f"{key}={_fmt(item)}")
    return "; ".join(parts[:4]) if parts else "-"


def _response_uses_matgl_columns(response: DiscoveryFullResponse) -> bool:
    if response.discovery_plan is not None:
        return response.discovery_plan.execution_policy.calculate_matgl
    return response.constraints.calculate_matgl
