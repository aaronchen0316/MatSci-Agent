from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    max_energy_above_hull: float = Field(default=0.1, ge=0)
    top_k: int = Field(default=5, ge=1, le=100)
    mp_filters: MPFilters = Field(default_factory=MPFilters)


class ParsedDiscoveryIntent(BaseModel):
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)
    requested_material_class: str = "unknown"


class DiscoveryRequest(BaseModel):
    research_goal: str = Field(min_length=5)
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)


class ExecutionPolicy(BaseModel):
    calculate_matgl: bool = False
    recalculate_top_n: int | None = Field(default=None, ge=1, le=100)
    matgl_max_recalc_entries: int = Field(default=10, ge=1, le=100)
    matgl_max_atoms: int = Field(default=50, ge=1, le=500)
    enable_relaxation: bool = False
    relaxation_max_steps: int = Field(default=200, ge=1, le=5000)


class DiscoveryPlan(BaseModel):
    research_goal_raw: str
    task_class: Literal[
        "band_gap_screening",
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
    band_gap_ev: float = Field(ge=0)
    uncertainty: float = Field(ge=0)
    backend: str


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


class DiscoveryResponse(BaseModel):
    research_goal: str
    constraints: DiscoveryConstraints
    status: Literal["success", "partial", "failed", "unsupported"]
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
    known_stability_present: bool | None = None


class CandidateBandGapSummary(BaseModel):
    material_id: str
    formula: str
    band_gap_ev: float = Field(ge=0)
    band_gap_source: str | None = None
    energy_above_hull: float | None = None
    is_stable: bool | None = None
    stability_source: str | None = None
    has_multiple_entries: bool = False
    entry_count: int = Field(default=1, ge=1)


class DiscoverySummaryResponse(BaseModel):
    status: Literal["success", "partial", "failed", "unsupported"] = "success"
    candidates: list[CandidateBandGapSummary] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    unsupported_reason: str | None = None


class MPRetrieverInput(BaseModel):
    research_goal: str
    constraints: DiscoveryConstraints
    exclude_material_ids: list[str] = Field(default_factory=list)
    limit_override: int | None = Field(default=None, ge=1, le=500)


class MPRetrieverOutput(BaseModel):
    candidates: list[Candidate]
    provenance: ToolCallProvenance


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
    recalculate_top_n: int | None = Field(default=None, ge=1, le=100)
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


class BandGapBenchmarkMetrics(BaseModel):
    sample_size: int = 0
    mae: float | None = None
    rmse: float | None = None
    rank_correlation: float | None = None
    failure_count: int = 0
    skipped_count: int = 0
    mp_passthrough_count: int = 0
    matgl_count: int = 0
    fallback_count: int = 0


class BandGapBenchmarkArtifact(BaseModel):
    mode: Literal["small", "full"]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: BandGapBenchmarkMetrics
    rows: list[dict[str, Any]] = Field(default_factory=list)
