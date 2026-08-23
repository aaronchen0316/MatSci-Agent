from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from multiagent.artifacts import HarnessArtifactStore
from multiagent.live_suite import scenario_assertion_failures
from multiagent.repair_validation import validate_repair_test_evidence
from multiagent.schemas import (
    CodexDebuggerInput,
    CodexDebuggerReport,
    FinalVerifierInput,
    FinalVerifierReport,
    HarnessAttemptRecord,
    HarnessRunReport,
    HarnessStopReason,
    LiveEvalEvidence,
    LiveEvalInput,
    LiveEvalScenario,
    MaterialsQueryCriticInput,
    MaterialsQueryCriticReport,
    RefreshFeedback,
    RetrievalTesterInput,
    RetrievalTesterReport,
    RepairTestEvidence,
)
from multiagent.settings import MultiAgentSettings
from multiagent.tools import ToolGroups, build_tool_groups, cleanup_worktree, run_scoped_live_evaluation, worktree_evidence

_MAX_REVIEW_CYCLES = 3


@dataclass(frozen=True)
class AgentRegistry:
    retrieval_tester: object
    materials_query_critic: object
    codex_debugger: object
    final_verifier: object


def _configure_sdk(settings: MultiAgentSettings) -> Any:
    try:
        sdk = importlib.import_module("agents")
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenAI Agents SDK not installed. Run `uv sync --extra dev --extra agents`.") from exc
    if settings.disable_tracing:
        sdk.set_tracing_disabled(disabled=True)
    if settings.api_key:
        client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        sdk.set_default_openai_client(client, use_for_tracing=not settings.disable_tracing)
    return sdk


def _agent_prompt(name: str, settings: MultiAgentSettings) -> str:
    return (settings.resolved_tool_root / "agent_specs" / f"{name}.md").read_text()


def _output_schema(sdk: Any, output_type: type[object]) -> object:
    return sdk.AgentOutputSchema(output_type, strict_json_schema=False)


def _build_agent_registry(sdk: Any, settings: MultiAgentSettings, tools: ToolGroups) -> AgentRegistry:
    return AgentRegistry(
        retrieval_tester=sdk.Agent(
            name="Retrieval Tester Agent",
            instructions=_agent_prompt("retrieval_tester", settings),
            model=settings.model,
            tools=tools.tester,
            output_type=_output_schema(sdk, RetrievalTesterReport),
        ),
        materials_query_critic=sdk.Agent(
            name="Materials Query Critic Agent",
            instructions=_agent_prompt("materials_query_critic", settings),
            model=settings.model,
            tools=tools.critic,
            output_type=_output_schema(sdk, MaterialsQueryCriticReport),
        ),
        codex_debugger=sdk.Agent(
            name="Codex Debugger Agent",
            instructions=_agent_prompt("codex_debugger", settings),
            model=settings.model,
            tools=tools.debugger,
            output_type=_output_schema(sdk, CodexDebuggerReport),
        ),
        final_verifier=sdk.Agent(
            name="Final Verifier Agent",
            instructions=_agent_prompt("final_verifier", settings),
            model=settings.model,
            tools=tools.verifier,
            output_type=_output_schema(sdk, FinalVerifierReport),
        ),
    )


@dataclass
class MultiAgentHarness:
    settings: MultiAgentSettings
    sdk: Any | None
    registry: AgentRegistry | None
    retrieval_tester_runner: Callable[[RetrievalTesterInput], Awaitable[RetrievalTesterReport]]
    materials_query_critic_runner: Callable[[MaterialsQueryCriticInput], Awaitable[MaterialsQueryCriticReport]]
    codex_debugger_runner: Callable[[CodexDebuggerInput], Awaitable[CodexDebuggerReport]]
    final_verifier_runner: Callable[[FinalVerifierInput], Awaitable[FinalVerifierReport]]
    runtime_rebinder: Callable[[Path], None] | None = None
    scenario_evaluator: Callable[[MultiAgentSettings, LiveEvalInput], LiveEvalEvidence] = run_scoped_live_evaluation
    repair_test_validator: Callable[[MultiAgentSettings, Path, list[str], list[str]], RepairTestEvidence] = (
        lambda settings, worktree_path, test_files, test_targets: validate_repair_test_evidence(
            settings,
            worktree_path,
            declared_test_files=test_files,
            declared_test_targets=test_targets,
        )
    )

    @classmethod
    def build(cls, settings: MultiAgentSettings | None = None) -> "MultiAgentHarness":
        runtime = settings or MultiAgentSettings.from_env()
        sdk = _configure_sdk(runtime)
        tool_groups = build_tool_groups(sdk, runtime)
        registry = _build_agent_registry(sdk, runtime, tool_groups)
        runner = sdk.Runner
        registry_holder = {"value": registry}

        async def run_retrieval_tester(payload: RetrievalTesterInput) -> RetrievalTesterReport:
            result = await runner.run(
                registry_holder["value"].retrieval_tester,
                payload.model_dump_json(indent=2),
                max_turns=runtime.max_agent_turns,
            )
            return RetrievalTesterReport.model_validate(result.final_output.model_dump(mode="json"))

        async def run_materials_query_critic(payload: MaterialsQueryCriticInput) -> MaterialsQueryCriticReport:
            result = await runner.run(
                registry_holder["value"].materials_query_critic,
                payload.model_dump_json(indent=2),
                max_turns=runtime.max_agent_turns,
            )
            return MaterialsQueryCriticReport.model_validate(result.final_output.model_dump(mode="json"))

        async def run_codex_debugger(payload: CodexDebuggerInput) -> CodexDebuggerReport:
            result = await runner.run(
                registry_holder["value"].codex_debugger,
                payload.model_dump_json(indent=2),
                max_turns=runtime.max_agent_turns,
            )
            return CodexDebuggerReport.model_validate(result.final_output.model_dump(mode="json"))

        async def run_final_verifier(payload: FinalVerifierInput) -> FinalVerifierReport:
            result = await runner.run(
                registry_holder["value"].final_verifier,
                payload.model_dump_json(indent=2),
                max_turns=runtime.max_agent_turns,
            )
            return FinalVerifierReport.model_validate(result.final_output.model_dump(mode="json"))

        harness = cls(
            settings=runtime,
            sdk=sdk,
            registry=registry,
            retrieval_tester_runner=run_retrieval_tester,
            materials_query_critic_runner=run_materials_query_critic,
            codex_debugger_runner=run_codex_debugger,
            final_verifier_runner=run_final_verifier,
        )

        def rebind_runtime(target_root: Path) -> None:
            scoped_settings = replace(runtime, active_target_root=target_root.resolve())
            scoped_tools = build_tool_groups(sdk, scoped_settings)
            scoped_registry = _build_agent_registry(sdk, scoped_settings, scoped_tools)
            registry_holder["value"] = scoped_registry
            harness.registry = scoped_registry

        harness.runtime_rebinder = rebind_runtime
        return harness

    async def repair_scenario(self, scenario: LiveEvalScenario) -> HarnessRunReport:
        """Run the fixed specialist loop for one failing live validation scenario."""

        objective = scenario.query
        store = HarnessArtifactStore.create(self.settings, scenario.name)
        attempts: list[HarnessAttemptRecord] = []
        refresh_feedback: RefreshFeedback | None = None
        branch_name: str | None = None
        worktree_path: str | None = None
        latest_tester: RetrievalTesterReport | None = None
        latest_critic: MaterialsQueryCriticReport | None = None
        latest_debugger: CodexDebuggerReport | None = None
        latest_repair_test_evidence: RepairTestEvidence | None = None
        latest_verifier: FinalVerifierReport | None = None
        active_target_root = self.settings.resolved_target_root

        def finish(
            *,
            status: str,
            stop_reason: HarnessStopReason,
            summary: str,
            next_step: str,
        ) -> HarnessRunReport:
            cleanup_status = "not_needed"
            if worktree_path:
                try:
                    store.write_json("worktree_evidence.json", worktree_evidence(self.settings, worktree_path))
                except ValueError as exc:
                    store.write_json("worktree_evidence.json", {"error": str(exc)})
                cleanup = cleanup_worktree(self.settings, worktree_path)
                cleanup_status = str(cleanup.get("status", "failed"))
                store.write_json("worktree_cleanup.json", cleanup)
            report = self._finalize_run_report(
                status=status,
                stop_reason=stop_reason,
                summary=summary,
                next_step=next_step,
                attempts=attempts,
                latest_tester=latest_tester,
                latest_critic=latest_critic,
                latest_debugger=latest_debugger,
                latest_repair_test_evidence=latest_repair_test_evidence,
                latest_verifier=latest_verifier,
                branch_name=branch_name,
                worktree_path=worktree_path,
                artifact_dir=str(store.run_dir),
                worktree_cleanup_status=cleanup_status,
            )
            store.write_model("harness_run_report.json", report)
            return report

        for attempt_number in range(1, _MAX_REVIEW_CYCLES + 1):
            tester_input = RetrievalTesterInput(
                objective=objective,
                refresh_feedback=refresh_feedback,
                allow_live_mp=True,
                live_evaluation_input=LiveEvalInput(
                    query=scenario.query,
                    constraints=scenario.constraints,
                    allow_live_mp=True,
                ),
                scenario_name=scenario.name,
            )
            store.write_model(f"attempts/{attempt_number}/retrieval_tester_input.json", tester_input)
            try:
                tester_report = await self.retrieval_tester_runner(tester_input)
            except Exception as exc:
                tester_report = RetrievalTesterReport(
                    status="blocked",
                    summary=f"Retrieval Tester execution failed: {type(exc).__name__}",
                )
            live_evidence = self.scenario_evaluator(
                replace(self.settings, active_target_root=active_target_root),
                tester_input.live_evaluation_input,
            )
            assertion_failures = scenario_assertion_failures(scenario, live_evidence)
            if live_evidence.query != scenario.query:
                assertion_failures.append("scoped evaluator query did not match requested scenario")
            provenance_blocked = "real Materials Project source was not used" in assertion_failures
            forced_status = "pass"
            forced_summary = tester_report.summary
            forced_stage = None
            if live_evidence.status == "blocked":
                forced_status = "blocked"
                forced_summary = live_evidence.blocked_reason or "required live scenario evaluation blocked"
                forced_stage = live_evidence.failed_stage or tester_report.failed_stage
            elif live_evidence.status == "fail" or (assertion_failures and not provenance_blocked):
                forced_status = "fail"
                forced_summary = "; ".join(assertion_failures) or f"live scenario failed at {live_evidence.failed_stage or 'unknown'}"
                forced_stage = live_evidence.failed_stage or tester_report.failed_stage
            elif provenance_blocked:
                forced_summary = "Live evaluator did not establish real Materials Project provenance."
            elif tester_report.status != "pass":
                forced_summary = "Scoped live scenario evaluation passed; tester status normalized to typed evaluator evidence."
            tester_report = tester_report.model_copy(
                update={
                    "status": forced_status,
                    "failed_stage": forced_stage,
                    "summary": forced_summary,
                    "live_evaluation": live_evidence,
                    "evidence": {
                        **tester_report.evidence,
                        "scenario_name": scenario.name,
                        "scenario_assertion_failures": assertion_failures,
                    },
                }
            )
            store.write_model(f"attempts/{attempt_number}/retrieval_tester_report.json", tester_report)
            latest_tester = tester_report
            attempt = HarnessAttemptRecord(
                attempt_number=attempt_number,
                branch_name=branch_name,
                worktree_path=worktree_path,
                refresh_feedback=refresh_feedback,
                tester_report=tester_report,
            )

            if tester_report.status == "blocked":
                attempt.stop_reason_fragment = "tester_blocked"
                attempts.append(attempt)
                return finish(
                    status="blocked",
                    stop_reason="tester_blocked",
                    summary="Retrieval tester blocked before repair loop could continue.",
                    next_step="Enable missing live-eval or repo prerequisites, then rerun harness.",
                )

            critic_input = MaterialsQueryCriticInput(
                objective=objective,
                tester_report=tester_report,
                review_evidence=tester_report.live_evaluation,
            )
            store.write_model(f"attempts/{attempt_number}/materials_query_critic_input.json", critic_input)
            try:
                critic_report = await self.materials_query_critic_runner(critic_input)
            except Exception as exc:
                critic_report = MaterialsQueryCriticReport(
                    verdict="blocked",
                    summary=f"Materials Query Critic execution failed: {type(exc).__name__}",
                    blocked_reason=f"agent execution failed: {type(exc).__name__}",
                )
            store.write_model(f"attempts/{attempt_number}/materials_query_critic_report.json", critic_report)
            latest_critic = critic_report
            attempt.critic_report = critic_report
            if critic_report.verdict == "blocked":
                attempt.stop_reason_fragment = "critic_blocked"
                attempts.append(attempt)
                return finish(
                    status="blocked",
                    stop_reason="critic_blocked",
                    summary="Materials Query Critic blocked; no safe independent review is available.",
                    next_step=critic_report.blocked_reason or "Unblock critic inputs, then rerun harness.",
                )

            if tester_report.status == "pass" and critic_report.verdict == "agree":
                if self._has_real_approval_evidence(tester_report):
                    attempt.stop_reason_fragment = "dual_review_pass"
                    attempts.append(attempt)
                    return finish(
                        status="pass",
                        stop_reason="dual_review_pass",
                        summary="Retrieval Tester and Materials Query Critic approved real Materials Project evidence.",
                        next_step="Review evaluator evidence or continue with broader eval coverage.",
                    )
                attempt.stop_reason_fragment = "scientific_evidence_blocked"
                attempts.append(attempt)
                return finish(
                    status="blocked",
                    stop_reason="scientific_evidence_blocked",
                    summary="Dual review could not certify a pass without real Materials Project candidate evidence.",
                    next_step="Enable live MP evaluation and attach its typed evaluator output to the tester report.",
                )

            if tester_report.status == "fail" and critic_report.verdict == "disagree":
                if attempt_number >= _MAX_REVIEW_CYCLES:
                    attempt.stop_reason_fragment = "review_cycle_exhausted"
                    attempts.append(attempt)
                    return finish(
                        status="fail",
                        stop_reason="review_cycle_exhausted",
                        summary="Critic requested a tester refresh, but the three-cycle review budget was exhausted.",
                        next_step="Inspect the reviewer disagreement and rerun if another evaluation cycle is justified.",
                    )
                attempt.stop_reason_fragment = "critic_disagreement_refresh"
                attempts.append(attempt)
                refresh_feedback = RefreshFeedback(
                    source="critic",
                    summary=critic_report.summary,
                    findings=critic_report.material_findings,
                )
                continue

            debugger_input = CodexDebuggerInput(
                tester_report=tester_report,
                critic_report=critic_report,
                target_branch_prefix=self.settings.repair_branch_prefix,
                existing_branch_name=branch_name,
                existing_worktree_path=worktree_path,
            )
            store.write_model(f"attempts/{attempt_number}/codex_debugger_input.json", debugger_input)
            try:
                debugger_report = await self.codex_debugger_runner(debugger_input)
            except Exception as exc:
                debugger_report = CodexDebuggerReport(
                    status="blocked",
                    change_summary=f"Codex Debugger execution failed: {type(exc).__name__}",
                )
            store.write_model(f"attempts/{attempt_number}/codex_debugger_report.json", debugger_report)
            latest_debugger = debugger_report
            branch_name = debugger_report.branch_name or branch_name
            worktree_path = debugger_report.worktree_path or worktree_path
            attempt.debugger_report = debugger_report
            attempt.branch_name = branch_name
            attempt.worktree_path = worktree_path
            if debugger_report.status == "blocked":
                attempt.stop_reason_fragment = "debugger_blocked"
                attempts.append(attempt)
                return finish(
                    status="blocked",
                    stop_reason="debugger_blocked",
                    summary="Codex Debugger blocked; repair could not proceed safely.",
                    next_step="Enable mutation prerequisites or resolve debugger blocker, then rerun harness.",
                )

            if not worktree_path:
                repair_test_evidence = RepairTestEvidence(status="blocked", issues=["debugger did not report a worktree path"])
            else:
                repair_test_evidence = self.repair_test_validator(
                    self.settings,
                    Path(worktree_path),
                    debugger_report.test_files,
                    debugger_report.test_targets,
                )
            latest_repair_test_evidence = repair_test_evidence
            attempt.repair_test_evidence = repair_test_evidence
            store.write_model(f"attempts/{attempt_number}/repair_test_evidence.json", repair_test_evidence)

            verifier_input = FinalVerifierInput(
                objective=objective,
                tester_report=tester_report,
                critic_report=critic_report,
                debugger_report=debugger_report,
                repair_test_evidence=repair_test_evidence,
            )
            store.write_model(f"attempts/{attempt_number}/final_verifier_input.json", verifier_input)
            try:
                verifier_report = await self.final_verifier_runner(verifier_input)
            except Exception as exc:
                verifier_report = FinalVerifierReport(
                    status="blocked",
                    summary=f"Final Verifier execution failed: {type(exc).__name__}",
                    requires_tester_refresh=False,
                )
            store.write_model(f"attempts/{attempt_number}/final_verifier_report.json", verifier_report)
            latest_verifier = verifier_report
            attempt.verifier_report = verifier_report

            if repair_test_evidence is not None and repair_test_evidence.status != "pass":
                attempt.stop_reason_fragment = "repair_test_evidence_failed"
                attempts.append(attempt)
                return finish(
                    status="fail",
                    stop_reason="repair_test_evidence_failed",
                    summary="Deterministic repair-test validation rejected the patch.",
                    next_step="Inspect retained branch and repair_test_evidence artifact.",
                )

            if verifier_report.status == "blocked":
                attempt.stop_reason_fragment = "verifier_blocked"
                attempts.append(attempt)
                return finish(
                    status="blocked",
                    stop_reason="verifier_blocked",
                    summary="Verifier blocked; repair loop could not reach final gate.",
                    next_step="Resolve verifier blocker, then rerun harness.",
                )
            if verifier_report.status == "fail":
                attempt.stop_reason_fragment = "verifier_fail"
                attempts.append(attempt)
                return finish(
                    status="fail",
                    stop_reason="verifier_fail",
                    summary="Verifier rejected repair loop outcome.",
                    next_step="Inspect retained branch and verifier review notes.",
                )
            if attempt_number >= _MAX_REVIEW_CYCLES:
                attempt.stop_reason_fragment = "review_cycle_exhausted"
                attempts.append(attempt)
                return finish(
                    status="fail",
                    stop_reason="review_cycle_exhausted",
                    summary="Patch review required a fresh tester and critic cycle, but the review budget was exhausted.",
                    next_step="Inspect attempt history and rerun harness if another repair cycle is justified.",
                )
            attempt.stop_reason_fragment = (
                "verifier_accepted_refresh" if verifier_report.status == "accepted" else "needs_tester_refresh"
            )
            attempts.append(attempt)
            refresh_feedback = RefreshFeedback(
                source="verifier",
                summary=verifier_report.tester_refresh_reason or verifier_report.summary,
                findings=list(verifier_report.review_notes),
            )
            if worktree_path:
                active_target_root = Path(worktree_path).resolve()
                if self.runtime_rebinder is not None:
                    self.runtime_rebinder(active_target_root)

        return finish(
            status="fail",
            stop_reason="review_cycle_exhausted",
            summary="Review loop exhausted retry budget.",
            next_step="Inspect attempt history and rerun harness if needed.",
        )

    @staticmethod
    def _has_real_approval_evidence(tester_report: RetrievalTesterReport) -> bool:
        evidence = tester_report.live_evaluation
        if evidence is None or evidence.status != "pass" or not evidence.real_source_used:
            return False
        snapshots = evidence.candidate_snapshots
        return bool(snapshots.raw and snapshots.filtered and snapshots.ranked)

    @staticmethod
    def _finalize_run_report(
        *,
        status: str,
        stop_reason: HarnessStopReason,
        summary: str,
        next_step: str,
        attempts: list[HarnessAttemptRecord],
        latest_tester: RetrievalTesterReport | None,
        latest_critic: MaterialsQueryCriticReport | None,
        latest_debugger: CodexDebuggerReport | None,
        latest_repair_test_evidence: RepairTestEvidence | None,
        latest_verifier: FinalVerifierReport | None,
        branch_name: str | None,
        worktree_path: str | None,
        artifact_dir: str,
        worktree_cleanup_status: str,
    ) -> HarnessRunReport:
        return HarnessRunReport(
            status=status,
            summary=summary,
            next_step=next_step,
            stop_reason=stop_reason,
            attempt_count=len(attempts),
            branch_name=branch_name,
            worktree_path=worktree_path,
            artifact_dir=artifact_dir,
            worktree_cleanup_status=worktree_cleanup_status,
            attempts=attempts,
            latest_tester_report=latest_tester,
            latest_critic_report=latest_critic,
            latest_debugger_report=latest_debugger,
            latest_repair_test_evidence=latest_repair_test_evidence,
            latest_verifier_report=latest_verifier,
        )
