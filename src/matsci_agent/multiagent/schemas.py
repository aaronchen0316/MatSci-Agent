from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalTesterInput(BaseModel):
    objective: str
    verifier_feedback: str | None = None
    allow_live_mp: bool = False


class RetrievalTesterReport(BaseModel):
    status: Literal["pass", "fail"]
    failed_stage: Literal[
        "intent_parse",
        "search_space_expansion",
        "mp_query_compilation",
        "mp_zero_results",
        "deterministic_filter",
        "llm_policy_filter",
        "ranking",
        "answer_format",
        "unknown",
    ] | None = None
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_debug_focus: list[str] = Field(default_factory=list)
    offline_commands: list[str] = Field(default_factory=list)
    live_commands: list[str] = Field(default_factory=list)


class MaterialsQueryCriticInput(BaseModel):
    tester_report: RetrievalTesterReport


class MaterialsQueryCriticReport(BaseModel):
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    owning_modules: list[str] = Field(default_factory=list)
    recommended_fix_order: list[str] = Field(default_factory=list)
    notes_for_debugger: list[str] = Field(default_factory=list)


class CodexDebuggerInput(BaseModel):
    tester_report: RetrievalTesterReport
    critic_report: MaterialsQueryCriticReport
    target_branch_prefix: str = "retrieval-fix"


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
