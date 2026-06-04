from __future__ import annotations

from dataclasses import dataclass

from matsci_agent.schemas import DiscoveryConstraints, DiscoveryRequest


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    request: DiscoveryRequest
    prereq_note: str | None = None
    enable_policy_filter: bool = False


SCENARIOS: dict[str, Scenario] = {
    "basic_success": Scenario(
        name="basic_success",
        description="Stable retrieval-first semiconductor shortlist.",
        request=DiscoveryRequest(
            research_goal="Find stable semiconductors without silicon and band gap above 2 eV",
            constraints=DiscoveryConstraints(
                banned_elements=["Si"],
                min_band_gap_ev=2.0,
                max_energy_above_hull=0.08,
                top_k=3,
            ),
        ),
    ),
    "policy_filter": Scenario(
        name="policy_filter",
        description="Show single LLM chemistry screening on a band-gap request.",
        request=DiscoveryRequest(
            research_goal="Find practical semiconductor materials without silicon and band gap above 2 eV",
            constraints=DiscoveryConstraints(
                banned_elements=["Si"],
                min_band_gap_ev=2.0,
                max_energy_above_hull=0.08,
                top_k=3,
            ),
        ),
        prereq_note="Needs remote LLM credentials because policy filter now fails closed without them.",
        enable_policy_filter=True,
    ),
    "unsupported_request": Scenario(
        name="unsupported_request",
        description="Exercise structured refusal for unsupported simulation requests.",
        request=DiscoveryRequest(
            research_goal="Estimate diffusivity in bulk materials with long molecular dynamics runs",
            constraints=DiscoveryConstraints(top_k=3),
        ),
    ),
    "matgl_recalc": Scenario(
        name="matgl_recalc",
        description="Force local MatGL recalculation path when structures and deps are available.",
        request=DiscoveryRequest(
            research_goal="Find semiconductor materials with recalculated band gaps",
            constraints=DiscoveryConstraints(
                calculate_matgl=True,
                min_band_gap_ev=1.0,
                top_k=3,
            ),
        ),
        prereq_note="Needs matgl/torch/dgl stack plus local model bundles.",
    ),
}


def get_scenario(name: str) -> Scenario | None:
    return SCENARIOS.get(name)
