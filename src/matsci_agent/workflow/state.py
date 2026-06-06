from __future__ import annotations

from typing import Any, TypedDict

from matsci_agent.schemas import (
    CapabilityAssessment,
    DiscoveryConstraints,
    DiscoveryPlan,
    RankedCandidate,
    ReportSummary,
)


class DiscoveryState(TypedDict, total=False):
    research_goal: str
    request_constraints: DiscoveryConstraints
    constraints: DiscoveryConstraints
    discovery_plan: DiscoveryPlan
    capability_assessment: CapabilityAssessment
    iteration: int
    max_iterations: int
    messages: list[str]
    raw_candidates: list[dict[str, Any]]
    filtered_candidates: list[dict[str, Any]]
    filter_records: list[dict[str, Any]]
    filter_replenish_attempts: int
    search_space_targets: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    ranked_candidates: list[RankedCandidate]
    stable_found: bool
    known_stability_present: bool
    report_summary: ReportSummary
    provenance: list[dict[str, Any]]
    status: str
