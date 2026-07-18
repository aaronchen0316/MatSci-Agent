from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from matsci_agent.multiagent.orchestrator import MultiAgentHarness
from matsci_agent.multiagent.schemas import (
    CodexDebuggerInput,
    CodexDebuggerReport,
    FinalVerifierInput,
    FinalVerifierReport,
    MaterialsQueryCriticInput,
    MaterialsQueryCriticReport,
    RetrievalTesterInput,
    RetrievalTesterReport,
)
from matsci_agent.multiagent.settings import MultiAgentSettings


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(
        repo_root=tmp_path,
        enable_live_mp=True,
        enable_git_write=True,
        worktree_root=tmp_path / "worktrees",
    )


def _tester_report(status: str, summary: str, failed_stage: str | None = None) -> RetrievalTesterReport:
    return RetrievalTesterReport(
        status=status,
        failed_stage=failed_stage,
        summary=summary,
        evidence={},
        recommended_debug_focus=[],
        offline_commands=[],
        live_commands=[],
    )


def _critic_report(status: str = "ready", blocked_reason: str | None = None) -> MaterialsQueryCriticReport:
    return MaterialsQueryCriticReport(
        status=status,
        root_cause="retriever issue" if status == "ready" else "critic blocked",
        confidence=0.9,
        owning_modules=["src/matsci_agent/tools/mp_retriever.py"],
        recommended_fix_order=["tighten filters"],
        notes_for_debugger=["patch compiler"],
        blocked_reason=blocked_reason,
    )


def _debugger_report(
    status: str = "patched",
    branch_name: str | None = "retrieval-fix-1",
    worktree_path: str | None = "/tmp/retrieval-fix-1",
) -> CodexDebuggerReport:
    return CodexDebuggerReport(
        status=status,
        branch_name=branch_name,
        worktree_path=worktree_path,
        files_touched=["src/matsci_agent/tools/mp_retriever.py"] if status != "blocked" else [],
        commit_sha="abc123" if status == "patched" else None,
        change_summary="patched filters" if status != "blocked" else "blocked",
        follow_up_for_verifier=["rerun tester"] if status == "patched" else [],
    )


def _verifier_report(status: str, summary: str, tester_refresh_reason: str | None = None) -> FinalVerifierReport:
    return FinalVerifierReport(
        status=status,
        summary=summary,
        requires_tester_refresh=status == "needs_tester_refresh",
        tester_refresh_reason=tester_refresh_reason,
        review_notes=[],
        acceptance_criteria=[],
    )


def _sequence_runner[T](results: Sequence[T], calls: list[object]):
    values = list(results)

    async def runner(payload):
        calls.append(payload)
        index = len(calls) - 1
        return values[index]

    return runner


def _make_harness(
    tmp_path: Path,
    *,
    tester_results: Sequence[RetrievalTesterReport],
    critic_results: Sequence[MaterialsQueryCriticReport] | None = None,
    debugger_results: Sequence[CodexDebuggerReport] | None = None,
    verifier_results: Sequence[FinalVerifierReport] | None = None,
    tester_calls: list[RetrievalTesterInput] | None = None,
    critic_calls: list[MaterialsQueryCriticInput] | None = None,
    debugger_calls: list[CodexDebuggerInput] | None = None,
    verifier_calls: list[FinalVerifierInput] | None = None,
) -> MultiAgentHarness:
    tester_log = tester_calls if tester_calls is not None else []
    critic_log = critic_calls if critic_calls is not None else []
    debugger_log = debugger_calls if debugger_calls is not None else []
    verifier_log = verifier_calls if verifier_calls is not None else []
    return MultiAgentHarness(
        settings=_settings(tmp_path),
        sdk=None,
        registry=None,
        retrieval_tester_runner=_sequence_runner(tester_results, tester_log),
        materials_query_critic_runner=_sequence_runner(critic_results or [], critic_log),
        codex_debugger_runner=_sequence_runner(debugger_results or [], debugger_log),
        final_verifier_runner=_sequence_runner(verifier_results or [], verifier_log),
    )


def test_orchestrator_stops_when_tester_passes(tmp_path: Path):
    tester_calls: list[RetrievalTesterInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("pass", "tester passed")],
        tester_calls=tester_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "pass"
    assert report.stop_reason == "tester_pass"
    assert report.attempt_count == 1
    assert len(tester_calls) == 1
    assert report.attempts[0].stop_reason_fragment == "tester_pass"
    assert report.latest_critic_report is None
    assert report.artifact_dir is not None
    artifact_dir = Path(report.artifact_dir)
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "attempts/1/retrieval_tester_input.json").is_file()
    assert (artifact_dir / "attempts/1/retrieval_tester_report.json").is_file()
    assert (artifact_dir / "harness_run_report.json").is_file()


def test_orchestrator_stops_when_tester_blocked(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("blocked", "live unavailable")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "blocked"
    assert report.stop_reason == "tester_blocked"
    assert report.attempt_count == 1
    assert report.attempts[0].critic_report is None


def test_orchestrator_runs_full_loop_then_stops_on_verifier_pass(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "zero hits", "mp_zero_results")],
        critic_results=[_critic_report()],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("pass", "looks good")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "pass"
    assert report.stop_reason == "verifier_pass"
    assert report.branch_name == "retrieval-fix-1"
    assert report.worktree_path == "/tmp/retrieval-fix-1"
    assert report.attempts[0].stop_reason_fragment == "verifier_pass"
    assert report.latest_debugger_report is not None


def test_orchestrator_reruns_tester_after_verifier_refresh(tmp_path: Path):
    tester_calls: list[RetrievalTesterInput] = []
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "first failure", "mp_zero_results"),
            _tester_report("pass", "second pass"),
        ],
        critic_results=[_critic_report()],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("needs_tester_refresh", "rerun tester", "verify refreshed behavior")],
        tester_calls=tester_calls,
        debugger_calls=debugger_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "pass"
    assert report.stop_reason == "tester_pass"
    assert report.attempt_count == 2
    assert report.attempts[0].stop_reason_fragment == "needs_tester_refresh"
    assert tester_calls[1].verifier_feedback == "verify refreshed behavior"
    assert debugger_calls[0].existing_branch_name is None
    assert report.branch_name == "retrieval-fix-1"


def test_orchestrator_fails_when_refresh_budget_exhausted(tmp_path: Path):
    tester_calls: list[RetrievalTesterInput] = []
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "fail 1", "mp_zero_results"),
            _tester_report("fail", "fail 2", "mp_zero_results"),
            _tester_report("fail", "fail 3", "mp_zero_results"),
        ],
        critic_results=[_critic_report(), _critic_report(), _critic_report()],
        debugger_results=[
            _debugger_report(branch_name="retrieval-fix-1", worktree_path="/tmp/retrieval-fix-1"),
            _debugger_report(branch_name=None, worktree_path=None),
            _debugger_report(branch_name=None, worktree_path=None),
        ],
        verifier_results=[
            _verifier_report("needs_tester_refresh", "retry 1", "feedback 1"),
            _verifier_report("needs_tester_refresh", "retry 2", "feedback 2"),
            _verifier_report("needs_tester_refresh", "retry 3", "feedback 3"),
        ],
        tester_calls=tester_calls,
        debugger_calls=debugger_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "fail"
    assert report.stop_reason == "verifier_refresh_exhausted"
    assert report.attempt_count == 3
    assert report.attempts[-1].stop_reason_fragment == "verifier_refresh_exhausted"
    assert tester_calls[1].verifier_feedback == "feedback 1"
    assert tester_calls[2].verifier_feedback == "feedback 2"
    assert debugger_calls[1].existing_branch_name == "retrieval-fix-1"
    assert debugger_calls[1].existing_worktree_path == "/tmp/retrieval-fix-1"
    assert debugger_calls[2].existing_branch_name == "retrieval-fix-1"
    assert report.branch_name == "retrieval-fix-1"


def test_orchestrator_stops_when_critic_blocked(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report(status="blocked", blocked_reason="missing evidence")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "blocked"
    assert report.stop_reason == "critic_blocked"
    assert report.next_step == "missing evidence"
    assert report.attempts[0].debugger_report is None


def test_orchestrator_stops_when_debugger_blocked(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report()],
        debugger_results=[_debugger_report(status="blocked", branch_name=None, worktree_path=None)],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "blocked"
    assert report.stop_reason == "debugger_blocked"
    assert report.latest_verifier_report is None


def test_orchestrator_stops_when_verifier_fails(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report()],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("fail", "patch not sufficient")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "fail"
    assert report.stop_reason == "verifier_fail"
    assert report.attempts[0].stop_reason_fragment == "verifier_fail"
