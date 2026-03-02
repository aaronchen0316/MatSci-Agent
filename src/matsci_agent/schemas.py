from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class DiscoveryConstraints(BaseModel):
    banned_elements: list[str] = Field(default_factory=list)
    required_elements: list[str] = Field(default_factory=list)
    min_band_gap_ev: float | None = Field(default=None, ge=0)
    calculate_matgl: bool = False
    max_energy_above_hull: float = Field(default=0.1, ge=0)
    top_k: int = Field(default=5, ge=1, le=100)


class DiscoveryRequest(BaseModel):
    research_goal: str = Field(min_length=5)
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)


class Candidate(BaseModel):
    material_id: str
    formula: str
    source: Literal["materials_project", "mock"] = "mock"
    features: dict[str, Any] = Field(default_factory=dict)


class PredictedProperties(BaseModel):
    band_gap_ev: float = Field(ge=0)
    uncertainty: float = Field(ge=0)
    backend: str


class StabilityResult(BaseModel):
    energy_above_hull: float
    is_stable: bool
    method: str


class RankedCandidate(BaseModel):
    rank: int
    candidate: Candidate
    predicted_properties: PredictedProperties
    stability: StabilityResult
    score: float
    provenance: dict[str, Any] = Field(default_factory=dict)


class DiscoveryResponse(BaseModel):
    research_goal: str
    constraints: DiscoveryConstraints
    status: Literal["success", "partial", "failed"]
    iterations: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidates: list[RankedCandidate] = Field(default_factory=list)
    provenance: list[ToolCallProvenance] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class CandidateBandGapSummary(BaseModel):
    material_id: str
    formula: str
    band_gap_ev: float = Field(ge=0)


class DiscoverySummaryResponse(BaseModel):
    candidates: list[CandidateBandGapSummary] = Field(default_factory=list)


class ToolCallProvenance(BaseModel):
    tool_name: str
    input_payload: dict[str, Any]
    output_summary: dict[str, Any] = Field(default_factory=dict)


class MPRetrieverInput(BaseModel):
    research_goal: str
    constraints: DiscoveryConstraints


class MPRetrieverOutput(BaseModel):
    candidates: list[Candidate]
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
