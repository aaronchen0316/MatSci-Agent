from __future__ import annotations

from typing import Any, TypedDict

from matsci_agent.schemas import DiscoveryConstraints, RankedCandidate


class DiscoveryState(TypedDict, total=False):
    research_goal: str
    constraints: DiscoveryConstraints
    iteration: int
    max_iterations: int
    messages: list[str]
    raw_candidates: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    ranked_candidates: list[RankedCandidate]
    stable_found: bool
    provenance: list[dict[str, Any]]
    status: str
