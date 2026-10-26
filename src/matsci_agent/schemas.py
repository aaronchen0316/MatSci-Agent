from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


DEFAULT_STABILITY_HULL_THRESHOLD = 0.1


class FloatRange(BaseModel):
    min: float | None = None
    max: float | None = None


class IntRange(BaseModel):
    min: int | None = None
    max: int | None = None


class MPFilters(BaseModel):
    formula: str | list[str] | None = None
    chemsys: str | list[str] | None = None
    material_ids: str | list[str] | None = None
    elements: list[str] = Field(default_factory=list)
    exclude_elements: list[str] = Field(default_factory=list)
    possible_species: list[str] = Field(default_factory=list)
    has_props: list[str] = Field(default_factory=list)
    is_metal: bool | None = None
    is_stable: bool | None = None
    is_gap_direct: bool | None = None
    theoretical: bool | None = None
    deprecated: bool | None = None
    has_reconstructed: bool | None = None
    crystal_system: str | None = None
    spacegroup_number: int | list[int] | None = None
    spacegroup_symbol: str | list[str] | None = None
    band_gap: FloatRange | None = None
    energy_above_hull: FloatRange | None = None
    formation_energy: FloatRange | None = None
    density: FloatRange | None = None
    efermi: FloatRange | None = None
    total_magnetization: FloatRange | None = None
    volume: FloatRange | None = None
    num_sites: IntRange | None = None
    num_elements: IntRange | None = None


class DiscoveryConstraints(BaseModel):
    banned_elements: list[str] = Field(default_factory=list)
    required_elements: list[str] = Field(default_factory=list)
    min_band_gap_ev: float | None = Field(default=None, ge=0)
    calculate_matgl: bool = False
    max_energy_above_hull: float | None = Field(default=None, ge=0)
    top_k: int = Field(default=5, ge=1, le=30)
    mp_filters: MPFilters = Field(default_factory=MPFilters)


class ParsedDiscoveryIntent(BaseModel):
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)
    requested_material_class: str = "unknown"


class DiscoveryRequest(BaseModel):
    research_goal: str = Field(min_length=5, max_length=500)
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)


class ExecutionPolicy(BaseModel):
    calculate_matgl: bool = False
    matgl_max_recalc_entries: int = Field(default=10, ge=1, le=100)
    matgl_max_atoms: int = Field(default=50, ge=1, le=500)
    enable_relaxation: bool = False
    relaxation_max_steps: int = Field(default=200, ge=1, le=5000)


class DiscoveryPlan(BaseModel):
    research_goal_raw: str
    task_class: Literal[
        "band_gap_screening",
        "mp_property_screening",
        "bulk_relaxation_only",
        "diffusivity_simulation",
        "molecular_dynamics",
        "transport_property_estimation",
        "defect_property_workflow",
        "general_ab_initio_simulation",
        "unknown_task",
    ] = "unknown_task"
    parsed_constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)
    application_intent: str = "unknown"
    source_universe: Literal["materials_project_entries", "unknown"] = "unknown"
    requested_material_class: str = "unknown"
    practicality_mode: str = "unknown"
    ranking_intent: Literal["band_gap_desc", "default"] = "default"
    reporting_focus: str = "compact_summary"
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)


class CapabilityAssessment(BaseModel):
    supported: bool
    reason_code: str | None = None
    reason_message: str | None = None
    closest_supported_mode: str | None = None
    suggested_next_action: str | None = None


class ReportSummary(BaseModel):
    scientific_summary: str
    execution_summary: str
    caveats: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    material_id: str
    formula: str
    source: Literal["materials_project", "mock"] = "mock"
    has_multiple_entries: bool = False
    entry_count: int = Field(default=1, ge=1)
    features: dict[str, Any] = Field(default_factory=dict)


class PredictedProperties(BaseModel):
    band_gap_ev: float | None = Field(default=None, ge=0)
    backend: str
    error: str | None = None


class StabilityResult(BaseModel):
    energy_above_hull: float | None = None
    is_stable: bool | None = None
    method: str
    source: Literal["materials_project", "unknown"] = "unknown"
    used_relaxation: bool = False


class RankedCandidate(BaseModel):
    rank: int
    candidate: Candidate
    predicted_properties: PredictedProperties
    stability: StabilityResult
    score: float
    provenance: dict[str, Any] = Field(default_factory=dict)


class ToolCallProvenance(BaseModel):
    tool_name: str
    input_payload: dict[str, Any]
    output_summary: dict[str, Any] = Field(default_factory=dict)


class SearchSpaceTarget(BaseModel):
    formula: str
    normalized_formula: str
    chemsys: str
    elements: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class SearchSpaceExpansionInput(BaseModel):
    research_goal: str
    discovery_plan: DiscoveryPlan
    target_count: int = Field(default=15, ge=1, le=30)


class SearchSpaceExpansionOutput(BaseModel):
    targets: list[SearchSpaceTarget] = Field(default_factory=list)
    provenance: ToolCallProvenance


class DiscoveryResponse(BaseModel):
    research_goal: str
    constraints: DiscoveryConstraints
    status: Literal["success", "partial", "failed", "unsupported"]
    reason_code: str | None = None
    data_source: str | None = None
    request_id: str | None = None
    iterations: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidates: list[RankedCandidate] = Field(default_factory=list)
    provenance: list[ToolCallProvenance] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    discovery_plan: DiscoveryPlan | None = None
    capability_assessment: CapabilityAssessment | None = None
    report_summary: ReportSummary | None = None


class DiscoveryFullResponse(DiscoveryResponse):
    raw_candidates: list[Candidate] = Field(default_factory=list)
    filtered_candidates: list[Candidate] = Field(default_factory=list)
    filter_records: list[PolicyFilterRecord] = Field(default_factory=list)
    search_space_targets: list[SearchSpaceTarget] = Field(default_factory=list)
    known_stability_present: bool | None = None


class CandidateBandGapSummary(BaseModel):
    material_id: str
    formula: str
    band_gap_ev: float | None = Field(default=None, ge=0)
    mp_band_gap_ev: float | None = Field(default=None, ge=0)
    matgl_band_gap_ev: float | None = Field(default=None, ge=0)
    band_gap_source: str | None = None
    matgl_error: str | None = None
    gap_difference_ev: float | None = None
    source: Literal["materials_project", "mock"] = "materials_project"
    energy_above_hull: float | None = None
    is_stable: bool | None = None
    stability_source: str | None = None
    has_multiple_entries: bool = False
    entry_count: int = Field(default=1, ge=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class DiscoverySummaryResponse(BaseModel):
    status: Literal["success", "partial", "failed", "unsupported"] = "success"
    reason_code: str | None = None
    data_source: str | None = None
    request_id: str | None = None
    candidates: list[CandidateBandGapSummary] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    unsupported_reason: str | None = None


def summary_from_result(result: DiscoveryResponse) -> DiscoverySummaryResponse:
    return DiscoverySummaryResponse(
        status=result.status,
        reason_code=result.reason_code,
        data_source=result.data_source,
        request_id=result.request_id,
        candidates=[
            CandidateBandGapSummary(
                material_id=item.candidate.material_id,
                formula=item.candidate.formula,
                band_gap_ev=item.predicted_properties.band_gap_ev,
                mp_band_gap_ev=item.candidate.features.get("mp_band_gap_ev"),
                matgl_band_gap_ev=item.candidate.features.get("matgl_band_gap_ev"),
                band_gap_source=item.candidate.features.get("band_gap_source"),
                matgl_error=item.candidate.features.get("matgl_error"),
                gap_difference_ev=item.candidate.features.get("gap_difference_ev"),
                source=item.candidate.source,
                energy_above_hull=item.stability.energy_above_hull,
                is_stable=item.stability.is_stable,
                stability_source=item.stability.source,
                has_multiple_entries=item.candidate.has_multiple_entries,
                entry_count=item.candidate.entry_count,
                properties=item.candidate.features.get("properties", {}),
            )
            for item in result.candidates
        ],
        messages=result.messages,
        unsupported_reason=(
            result.capability_assessment.reason_message
            if result.capability_assessment is not None and not result.capability_assessment.supported
            else None
        ),
    )


class MPRetrieverInput(BaseModel):
    research_goal: str
    constraints: DiscoveryConstraints
    exclude_material_ids: list[str] = Field(default_factory=list)
    limit_override: int | None = Field(default=None, ge=1, le=500)
    search_space_targets: list[SearchSpaceTarget] = Field(default_factory=list)


class MPRetrieverOutput(BaseModel):
    candidates: list[Candidate]
    provenance: ToolCallProvenance
    reason_code: str | None = None


class PolicyFilterInput(BaseModel):
    candidates: list[Candidate]
    discovery_plan: DiscoveryPlan


class PolicyFilterDecisionPayload(BaseModel):
    material_id: str
    keep: bool
    reasons: list[str] = Field(default_factory=list)


class PolicyFilterBatchResponse(BaseModel):
    policy_name: str
    decisions: list[PolicyFilterDecisionPayload] = Field(default_factory=list)


class PolicyFilterRecord(BaseModel):
    candidate: Candidate
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    policy: str


class PolicyFilterOutput(BaseModel):
    filtered_candidates: list[Candidate]
    records: list[PolicyFilterRecord]
    provenance: ToolCallProvenance


class PropertyPredictorInput(BaseModel):
    candidates: list[Candidate]
    goal: str
    calculate_matgl: bool = False
    matgl_max_recalc_entries: int = Field(default=10, ge=1, le=100)
    matgl_max_atoms: int = Field(default=50, ge=1, le=500)
    enable_relaxation: bool = False
    relaxation_max_steps: int = Field(default=200, ge=1, le=5000)


class PropertyPredictionRecord(BaseModel):
    candidate: Candidate
    predicted: PredictedProperties


class PropertyPredictorOutput(BaseModel):
    predictions: list[PropertyPredictionRecord]
    provenance: ToolCallProvenance


class StabilityCheckerInput(BaseModel):
    predictions: list[PropertyPredictionRecord]
    constraints: DiscoveryConstraints


class StabilityRecord(BaseModel):
    candidate: Candidate
    predicted: PredictedProperties
    stability: StabilityResult


class StabilityCheckerOutput(BaseModel):
    records: list[StabilityRecord]
    provenance: ToolCallProvenance
