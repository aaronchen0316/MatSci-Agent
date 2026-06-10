from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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
    "tester_pass",
    "tester_blocked",
    "critic_blocked",
    "debugger_blocked",
    "verifier_pass",
    "verifier_fail",
    "verifier_blocked",
    "verifier_refresh_exhausted",
]


class RetrievalTesterInput(BaseModel):
    objective: str
    verifier_feedback: str | None = None
    allow_live_mp: bool = False


class LiveEvalInput(BaseModel):
    query: str
    allow_live_mp: bool = False


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


class CompiledFilterEvidence(BaseModel):
    effective_filters: dict[str, Any] = Field(default_factory=dict)
    mp_search_kwargs: dict[str, Any] = Field(default_factory=dict)
    mp_search_kwargs_sequence: list[dict[str, Any]] = Field(default_factory=list)


class LiveEvalEvidence(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    query: str
    failed_stage: FailureStage | None = None
    compiled_filters: CompiledFilterEvidence = Field(default_factory=CompiledFilterEvidence)
    result_counts: StageCounts = Field(default_factory=StageCounts)
    constraint_violations: ConstraintViolationSummary = Field(default_factory=ConstraintViolationSummary)
    messages: list[str] = Field(default_factory=list)
    provenance_summary: dict[str, Any] = Field(default_factory=dict)
    real_source_used: bool = False
    blocked_reason: str | None = None


class RetrievalTesterReport(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    failed_stage: FailureStage | None = None
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_debug_focus: list[str] = Field(default_factory=list)
    offline_commands: list[str] = Field(default_factory=list)
    live_commands: list[str] = Field(default_factory=list)


class MaterialsQueryCriticInput(BaseModel):
    tester_report: RetrievalTesterReport


class MaterialsQueryCriticReport(BaseModel):
    status: Literal["ready", "blocked"] = "ready"
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    owning_modules: list[str] = Field(default_factory=list)
    recommended_fix_order: list[str] = Field(default_factory=list)
    notes_for_debugger: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None


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
    pr_url: str | None = None
    change_summary: str
    follow_up_for_verifier: list[str] = Field(default_factory=list)


class FinalVerifierInput(BaseModel):
    objective: str
    tester_report: RetrievalTesterReport
    critic_report: MaterialsQueryCriticReport
    debugger_report: CodexDebuggerReport


class FinalVerifierReport(BaseModel):
    status: Literal["pass", "fail", "needs_tester_refresh", "blocked"]
    summary: str
    requires_tester_refresh: bool = False
    tester_refresh_reason: str | None = None
    review_notes: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class ControllerSummary(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    summary: str
    next_step: str
    branch_name: str | None = None
    pr_url: str | None = None


class HarnessAttemptRecord(BaseModel):
    attempt_number: int = Field(ge=1)
    branch_name: str | None = None
    worktree_path: str | None = None
    pr_url: str | None = None
    tester_report: RetrievalTesterReport
    critic_report: MaterialsQueryCriticReport | None = None
    debugger_report: CodexDebuggerReport | None = None
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
    pr_url: str | None = None
    attempts: list[HarnessAttemptRecord] = Field(default_factory=list)
    latest_tester_report: RetrievalTesterReport | None = None
    latest_critic_report: MaterialsQueryCriticReport | None = None
    latest_debugger_report: CodexDebuggerReport | None = None
    latest_verifier_report: FinalVerifierReport | None = None
