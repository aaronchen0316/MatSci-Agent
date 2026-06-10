from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from matsci_agent.multiagent.factory import AgentRegistry, build_agent_registry
from matsci_agent.multiagent.schemas import (
    CodexDebuggerInput,
    CodexDebuggerReport,
    FinalVerifierInput,
    FinalVerifierReport,
    HarnessAttemptRecord,
    HarnessRunReport,
    HarnessStopReason,
    MaterialsQueryCriticInput,
    MaterialsQueryCriticReport,
    RetrievalTesterInput,
    RetrievalTesterReport,
)
from matsci_agent.multiagent.sdk import configure_sdk
from matsci_agent.multiagent.settings import MultiAgentSettings
from matsci_agent.multiagent.tools import build_tool_groups

_MAX_REPAIR_ATTEMPTS = 3


@dataclass
class MultiAgentHarness:
    settings: MultiAgentSettings
    sdk: Any | None
    registry: AgentRegistry | None
    retrieval_tester_runner: Callable[[RetrievalTesterInput], Awaitable[RetrievalTesterReport]]
    materials_query_critic_runner: Callable[[MaterialsQueryCriticInput], Awaitable[MaterialsQueryCriticReport]]
    codex_debugger_runner: Callable[[CodexDebuggerInput], Awaitable[CodexDebuggerReport]]
    final_verifier_runner: Callable[[FinalVerifierInput], Awaitable[FinalVerifierReport]]

    @classmethod
    def build(cls, settings: MultiAgentSettings | None = None) -> "MultiAgentHarness":
        runtime = settings or MultiAgentSettings.from_env()
        sdk = configure_sdk(runtime)
        tool_groups = build_tool_groups(sdk, runtime)

        # We wrap sub-agents as function tools here instead of using Agent.as_tool()
        # directly. This gives us explicit control over:
        # - input schemas
        # - typed final-output serialization
        # - future per-agent logging / branch policy hooks
        registry = build_agent_registry(sdk, runtime, tool_groups, controller_tools=[])
        runner = sdk.Runner

        async def run_retrieval_tester(payload: RetrievalTesterInput) -> RetrievalTesterReport:
            result = await runner.run(
                registry.retrieval_tester,
                payload.model_dump_json(indent=2),
            )
            return RetrievalTesterReport.model_validate(result.final_output.model_dump(mode="json"))

        async def run_materials_query_critic(payload: MaterialsQueryCriticInput) -> MaterialsQueryCriticReport:
            result = await runner.run(
                registry.materials_query_critic,
                payload.model_dump_json(indent=2),
            )
            return MaterialsQueryCriticReport.model_validate(result.final_output.model_dump(mode="json"))

        async def run_codex_debugger(payload: CodexDebuggerInput) -> CodexDebuggerReport:
            result = await runner.run(
                registry.codex_debugger,
                payload.model_dump_json(indent=2),
            )
            return CodexDebuggerReport.model_validate(result.final_output.model_dump(mode="json"))

        async def run_final_verifier(payload: FinalVerifierInput) -> FinalVerifierReport:
            result = await runner.run(
                registry.final_verifier,
                payload.model_dump_json(indent=2),
            )
            return FinalVerifierReport.model_validate(result.final_output.model_dump(mode="json"))

        return cls(
            settings=runtime,
            sdk=sdk,
            registry=registry,
            retrieval_tester_runner=run_retrieval_tester,
            materials_query_critic_runner=run_materials_query_critic,
            codex_debugger_runner=run_codex_debugger,
            final_verifier_runner=run_final_verifier,
        )

    async def run(self, objective: str) -> HarnessRunReport:
        attempts: list[HarnessAttemptRecord] = []
        verifier_feedback: str | None = None
        branch_name: str | None = None
        worktree_path: str | None = None
        pr_url: str | None = None
        latest_tester: RetrievalTesterReport | None = None
        latest_critic: MaterialsQueryCriticReport | None = None
        latest_debugger: CodexDebuggerReport | None = None
        latest_verifier: FinalVerifierReport | None = None

        for attempt_number in range(1, _MAX_REPAIR_ATTEMPTS + 1):
            tester_report = await self.retrieval_tester_runner(
                RetrievalTesterInput(
                    objective=objective,
                    verifier_feedback=verifier_feedback,
                    allow_live_mp=self.settings.enable_live_mp,
                )
            )
            latest_tester = tester_report
            attempt = HarnessAttemptRecord(
                attempt_number=attempt_number,
                branch_name=branch_name,
                worktree_path=worktree_path,
                pr_url=pr_url,
                tester_report=tester_report,
            )

            if tester_report.status == "pass":
                attempt.stop_reason_fragment = "tester_pass"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="pass",
                    stop_reason="tester_pass",
                    summary="Retrieval tester passed; no repair loop needed.",
                    next_step="Review evaluator evidence or continue with broader eval coverage.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            if tester_report.status == "blocked":
                attempt.stop_reason_fragment = "tester_blocked"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="blocked",
                    stop_reason="tester_blocked",
                    summary="Retrieval tester blocked before repair loop could continue.",
                    next_step="Enable missing live-eval or repo prerequisites, then rerun harness.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            critic_report = await self.materials_query_critic_runner(
                MaterialsQueryCriticInput(tester_report=tester_report)
            )
            latest_critic = critic_report
            attempt.critic_report = critic_report

            if critic_report.status == "blocked":
                attempt.stop_reason_fragment = "critic_blocked"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="blocked",
                    stop_reason="critic_blocked",
                    summary="Materials Query Critic blocked; no safe root-cause diagnosis available.",
                    next_step=critic_report.blocked_reason or "Unblock critic inputs, then rerun harness.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            debugger_report = await self.codex_debugger_runner(
                CodexDebuggerInput(
                    tester_report=tester_report,
                    critic_report=critic_report,
                    target_branch_prefix="retrieval-fix",
                    existing_branch_name=branch_name,
                    existing_worktree_path=worktree_path,
                )
            )
            latest_debugger = debugger_report
            branch_name = debugger_report.branch_name or branch_name
            worktree_path = debugger_report.worktree_path or worktree_path
            pr_url = debugger_report.pr_url or pr_url
            attempt.debugger_report = debugger_report
            attempt.branch_name = branch_name
            attempt.worktree_path = worktree_path
            attempt.pr_url = pr_url

            if debugger_report.status == "blocked":
                attempt.stop_reason_fragment = "debugger_blocked"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="blocked",
                    stop_reason="debugger_blocked",
                    summary="Codex Debugger blocked; repair could not proceed safely.",
                    next_step="Enable mutation prerequisites or relax blocked condition, then rerun harness.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            verifier_report = await self.final_verifier_runner(
                FinalVerifierInput(
                    objective=objective,
                    tester_report=tester_report,
                    critic_report=critic_report,
                    debugger_report=debugger_report,
                )
            )
            latest_verifier = verifier_report
            attempt.verifier_report = verifier_report

            if verifier_report.status == "pass":
                attempt.stop_reason_fragment = "verifier_pass"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="pass",
                    stop_reason="verifier_pass",
                    summary="Verifier accepted repair loop outcome.",
                    next_step="Review diff or PR, then merge when ready.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            if verifier_report.status == "blocked":
                attempt.stop_reason_fragment = "verifier_blocked"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="blocked",
                    stop_reason="verifier_blocked",
                    summary="Verifier blocked; repair loop could not reach final gate.",
                    next_step="Resolve verifier blocker, then rerun harness.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            if verifier_report.status == "fail":
                attempt.stop_reason_fragment = "verifier_fail"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="fail",
                    stop_reason="verifier_fail",
                    summary="Verifier rejected repair loop outcome.",
                    next_step="Inspect verifier review notes and retry manually if still worthwhile.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )

            if attempt_number >= _MAX_REPAIR_ATTEMPTS:
                attempt.stop_reason_fragment = "verifier_refresh_exhausted"
                attempts.append(attempt)
                return self._finalize_run_report(
                    status="fail",
                    stop_reason="verifier_refresh_exhausted",
                    summary="Verifier requested tester refresh, but retry budget was exhausted.",
                    next_step="Inspect attempt history and rerun harness if more repair rounds are justified.",
                    attempts=attempts,
                    latest_tester=latest_tester,
                    latest_critic=latest_critic,
                    latest_debugger=latest_debugger,
                    latest_verifier=latest_verifier,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    pr_url=pr_url,
                )
            attempt.stop_reason_fragment = "needs_tester_refresh"
            attempts.append(attempt)
            verifier_feedback = verifier_report.tester_refresh_reason or verifier_report.summary

        return self._finalize_run_report(
            status="fail",
            stop_reason="verifier_refresh_exhausted",
            summary="Repair loop exhausted retry budget.",
            next_step="Inspect attempt history and rerun harness if needed.",
            attempts=attempts,
            latest_tester=latest_tester,
            latest_critic=latest_critic,
            latest_debugger=latest_debugger,
            latest_verifier=latest_verifier,
            branch_name=branch_name,
            worktree_path=worktree_path,
            pr_url=pr_url,
        )

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
        latest_verifier: FinalVerifierReport | None,
        branch_name: str | None,
        worktree_path: str | None,
        pr_url: str | None,
    ) -> HarnessRunReport:
        return HarnessRunReport(
            status=status,
            summary=summary,
            next_step=next_step,
            stop_reason=stop_reason,
            attempt_count=len(attempts),
            branch_name=branch_name,
            worktree_path=worktree_path,
            pr_url=pr_url,
            attempts=attempts,
            latest_tester_report=latest_tester,
            latest_critic_report=latest_critic,
            latest_debugger_report=latest_debugger,
            latest_verifier_report=latest_verifier,
        )
