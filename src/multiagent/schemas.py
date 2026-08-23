from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from matsci_agent.schemas import DiscoveryConstraints

FailureStage = Literal[
    "intent_parse",
    "search_space_expansion",
    "mp_query_compilation",
    "mp_zero_results",
    "deterministic_filter",
    "llm_policy_filter",
    "ranking",
    "answer_format",
    "unknown",
]

HarnessStopReason = Literal[
    "dual_review_pass",
    "tester_blocked",
    "critic_blocked",
    "scientific_evidence_blocked",
    "debugger_blocked",
    "verifier_fail",
    "verifier_blocked",
    "repair_test_evidence_failed",
    "review_cycle_exhausted",
]


class RefreshFeedback(BaseModel):
    source: Literal["critic", "verifier"]
    summary: str
    findings: list[str] = Field(default_factory=list)


class LiveEvalInput(BaseModel):
    query: str
    constraints: DiscoveryConstraints | None = None
    allow_live_mp: bool = False


class RetrievalTesterInput(BaseModel):
    objective: str
    refresh_feedback: RefreshFeedback | None = None
    allow_live_mp: bool = False
    live_evaluation_input: LiveEvalInput | None = None
    scenario_name: str | None = None


class ConstraintViolationRecord(BaseModel):
    stage: Literal["raw", "filtered", "ranked"]
    material_id: str
    formula: str
    violations: list[str] = Field(default_factory=list)


class ConstraintViolationSummary(BaseModel):
    raw: list[ConstraintViolationRecord] = Field(default_factory=list)
    filtered: list[ConstraintViolationRecord] = Field(default_factory=list)
    ranked: list[ConstraintViolationRecord] = Field(default_factory=list)


class StageCounts(BaseModel):
    raw_count: int = 0
    filtered_count: int = 0
    ranked_count: int = 0
    search_space_target_count: int = 0
    replenish_count: int = 0


class CandidateReviewSnapshot(BaseModel):
    """Bounded, chemistry-relevant candidate evidence for Critic review."""

    material_id: str
    formula: str
    elements: list[str] = Field(default_factory=list)
    mp_band_gap_ev: float | None = None
    energy_above_hull: float | None = None
    formation_energy: float | None = None
    density: float | None = None
    volume: float | None = None
    num_sites: int | None = None
    is_metal: bool | None = None
    is_stable: bool | None = None
    crystal_system: str | None = None
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None
    policy_passed: bool | None = None
    policy_reasons: list[str] = Field(default_factory=list)
    rank: int | None = None
    score: float | None = None
    predicted_band_gap_ev: float | None = None
    stability_energy_above_hull: float | None = None


class CandidateReviewSnapshots(BaseModel):
    raw: list[CandidateReviewSnapshot] = Field(default_factory=list, max_length=20)
    filtered: list[CandidateReviewSnapshot] = Field(default_factory=list, max_length=20)
    ranked: list[CandidateReviewSnapshot] = Field(default_factory=list)


class CompiledFilterEvidence(BaseModel):
    effective_filters: dict[str, Any] = Field(default_factory=dict)
    mp_search_kwargs: dict[str, Any] = Field(default_factory=dict)
    mp_search_kwargs_sequence: list[dict[str, Any]] = Field(default_factory=list)
    missing_filter_keys: list[str] = Field(default_factory=list)


class LiveEvalEvidence(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    query: str
    failed_stage: FailureStage | None = None
    compiled_filters: CompiledFilterEvidence = Field(default_factory=CompiledFilterEvidence)
    result_counts: StageCounts = Field(default_factory=StageCounts)
    constraint_violations: ConstraintViolationSummary = Field(default_factory=ConstraintViolationSummary)
    messages: list[str] = Field(default_factory=list)
    provenance_summary: dict[str, Any] = Field(default_factory=dict)
    candidate_snapshots: CandidateReviewSnapshots = Field(default_factory=CandidateReviewSnapshots)
    real_source_used: bool = False
    blocked_reason: str | None = None


class RetrievalTesterReport(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    failed_stage: FailureStage | None = None
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    live_evaluation: LiveEvalEvidence | None = None
    recommended_debug_focus: list[str] = Field(default_factory=list)
    offline_commands: list[str] = Field(default_factory=list)
    live_commands: list[str] = Field(default_factory=list)


class MaterialsQueryCriticInput(BaseModel):
    objective: str
    tester_report: RetrievalTesterReport
    review_evidence: LiveEvalEvidence | None = None


class MaterialsQueryCriticReport(BaseModel):
    verdict: Literal["agree", "disagree", "blocked"]
    summary: str
    material_findings: list[str] = Field(default_factory=list)
    owning_modules: list[str] = Field(default_factory=list)
    recommended_fix_order: list[str] = Field(default_factory=list)
    notes_for_debugger: list[str] = Field(default_factory=list)
    informational_notes: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_agreement_notes(cls, value: object) -> object:
        """Normalize internally contradictory Critic verdicts before validation."""

        if not isinstance(value, dict) or value.get("verdict") != "agree":
            return value
        extra_notes = [
            *(value.get("material_findings") or []),
            *(value.get("notes_for_debugger") or []),
            *(value.get("recommended_fix_order") or []),
        ]
        if not extra_notes and not value.get("owning_modules"):
            return value
        normalized = dict(value)
        module_notes = [f"Critic referenced module: {module}" for module in (value.get("owning_modules") or [])]
        normalized["material_findings"] = []
        normalized["owning_modules"] = []
        normalized["recommended_fix_order"] = []
        normalized["notes_for_debugger"] = []
        normalized["informational_notes"] = [
            *(normalized.get("informational_notes") or []),
            *extra_notes,
            *module_notes,
        ]
        return normalized

    @model_validator(mode="after")
    def validate_verdict_fields(self) -> "MaterialsQueryCriticReport":
        if self.verdict == "agree":
            if self.material_findings:
                raise ValueError("agree verdict must not include material findings")
            if self.owning_modules or self.recommended_fix_order or self.notes_for_debugger:
                raise ValueError("agree verdict must not include repair guidance")
            if self.blocked_reason is not None:
                raise ValueError("agree verdict must not include blocked_reason")
        elif self.verdict == "disagree":
            if not self.material_findings:
                raise ValueError("disagree verdict requires material findings")
            if self.blocked_reason is not None:
                raise ValueError("disagree verdict must not include blocked_reason")
        else:
            if not self.blocked_reason:
                raise ValueError("blocked verdict requires blocked_reason")
            if self.material_findings or self.owning_modules or self.recommended_fix_order or self.notes_for_debugger:
                raise ValueError("blocked verdict must not include findings or repair guidance")
        return self


class CodexDebuggerInput(BaseModel):
    tester_report: RetrievalTesterReport
    critic_report: MaterialsQueryCriticReport
    target_branch_prefix: str = "retrieval-fix"
    existing_branch_name: str | None = None
    existing_worktree_path: str | None = None


class CodexDebuggerReport(BaseModel):
    status: Literal["no_change", "patched", "blocked"]
    branch_name: str | None = None
    worktree_path: str | None = None
    files_touched: list[str] = Field(default_factory=list)
    commit_sha: str | None = None
    test_files: list[str] = Field(default_factory=list)
    test_targets: list[str] = Field(default_factory=list)
    change_summary: str
    follow_up_for_verifier: list[str] = Field(default_factory=list)


class RepairTestEvidence(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    changed_source_files: list[str] = Field(default_factory=list)
    changed_test_files: list[str] = Field(default_factory=list)
    deleted_or_renamed_test_files: list[str] = Field(default_factory=list)
    declared_test_files: list[str] = Field(default_factory=list)
    declared_test_targets: list[str] = Field(default_factory=list)
    targeted_test_output: str = ""
    full_test_output: str = ""
    coverage_before: dict[str, float] = Field(default_factory=dict)
    coverage_after: dict[str, float] = Field(default_factory=dict)
    coverage_regressions: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class FinalVerifierInput(BaseModel):
    objective: str
    tester_report: RetrievalTesterReport
    critic_report: MaterialsQueryCriticReport
    debugger_report: CodexDebuggerReport
    repair_test_evidence: RepairTestEvidence | None = None


class FinalVerifierReport(BaseModel):
    status: Literal["accepted", "fail", "needs_tester_refresh", "blocked"]
    summary: str
    requires_tester_refresh: bool = False
    tester_refresh_reason: str | None = None
    review_notes: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_refresh_requirement(self) -> "FinalVerifierReport":
        refresh_required = self.status in {"accepted", "needs_tester_refresh"}
        if self.requires_tester_refresh != refresh_required:
            raise ValueError("requires_tester_refresh must match verifier status")
        return self


class HarnessAttemptRecord(BaseModel):
    attempt_number: int = Field(ge=1)
    branch_name: str | None = None
    worktree_path: str | None = None
    refresh_feedback: RefreshFeedback | None = None
    tester_report: RetrievalTesterReport
    critic_report: MaterialsQueryCriticReport | None = None
    debugger_report: CodexDebuggerReport | None = None
    repair_test_evidence: RepairTestEvidence | None = None
    verifier_report: FinalVerifierReport | None = None
    stop_reason_fragment: str | None = None


class HarnessRunReport(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    summary: str
    next_step: str
    stop_reason: HarnessStopReason
    attempt_count: int = Field(ge=1)
    branch_name: str | None = None
    worktree_path: str | None = None
    artifact_dir: str | None = None
    worktree_cleanup_status: str | None = None
    attempts: list[HarnessAttemptRecord] = Field(default_factory=list)
    latest_tester_report: RetrievalTesterReport | None = None
    latest_critic_report: MaterialsQueryCriticReport | None = None
    latest_debugger_report: CodexDebuggerReport | None = None
    latest_repair_test_evidence: RepairTestEvidence | None = None
    latest_verifier_report: FinalVerifierReport | None = None


class LiveEvalScenario(BaseModel):
    name: str
    query: str
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)
    min_ranked_count: int = Field(default=1, ge=1)
    require_target_quality: bool = False


class LiveEvalScenarioResult(BaseModel):
    scenario: LiveEvalScenario
    evidence: LiveEvalEvidence
    assertion_failures: list[str] = Field(default_factory=list)


class LiveEvalSuiteReport(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    artifact_dir: str | None = None
    scenarios: list[LiveEvalScenarioResult] = Field(default_factory=list)


class PullRequestPublication(BaseModel):
    status: Literal["merged", "published", "blocked", "failed"]
    branch_name: str
    base_branch: str
    summary: str
    artifact_dir: str | None = None
    local_ci_output: str = ""
    pull_request_number: int | None = None
    pull_request_url: str | None = None
    head_sha: str | None = None
    merge_sha: str | None = None
    ci_status: Literal["pass", "fail", "timeout"] | None = None


class ModelPreflightReport(BaseModel):
    status: Literal["pass", "fallback", "blocked"]
    primary_model: str
    selected_model: str | None = None
    selected_product_model: str | None = None
    attempts: list[str] = Field(default_factory=list)
    summary: str


class ValidationRepairAttempt(BaseModel):
    scenario_name: str
    harness_report: HarnessRunReport | None = None
    publication: PullRequestPublication | None = None
    summary: str


class AdoptedBranchAttempt(BaseModel):
    branch_name: str
    status: Literal["merged", "rejected", "deleted", "blocked", "failed"]
    summary: str
    scenario_name: str | None = None
    artifact_dir: str | None = None
    rebased_from_sha: str | None = None
    harness_report: HarnessRunReport | None = None
    publication: PullRequestPublication | None = None


class ValidationRepairReport(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    summary: str
    artifact_dir: str | None = None
    model_preflight: ModelPreflightReport
    baseline: LiveEvalSuiteReport | None = None
    adopted_branches: list[AdoptedBranchAttempt] = Field(default_factory=list)
    attempts: list[ValidationRepairAttempt] = Field(default_factory=list)
    final: LiveEvalSuiteReport | None = None
