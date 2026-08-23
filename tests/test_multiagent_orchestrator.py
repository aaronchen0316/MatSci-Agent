from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from matsci_agent.multiagent.orchestrator import MultiAgentHarness
from matsci_agent.multiagent.live_suite import get_live_scenario
from matsci_agent.multiagent.schemas import (
    CandidateReviewSnapshot,
    CandidateReviewSnapshots,
    CodexDebuggerInput,
    CodexDebuggerReport,
    FinalVerifierInput,
    FinalVerifierReport,
    LiveEvalEvidence,
    MaterialsQueryCriticInput,
    MaterialsQueryCriticReport,
    RefreshFeedback,
    RetrievalTesterInput,
    RetrievalTesterReport,
    RepairTestEvidence,
    StageCounts,
)
from matsci_agent.multiagent.settings import MultiAgentSettings


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(
        repo_root=tmp_path,
        enable_live_mp=True,
        enable_git_write=True,
        worktree_root=tmp_path / "worktrees",
    )


def _live_evidence(*, real_source_used: bool = True, include_snapshots: bool = True) -> LiveEvalEvidence:
    snapshot = CandidateReviewSnapshot(material_id="mp-1", formula="TiO2", elements=["O", "Ti"], mp_band_gap_ev=3.0)
    snapshots = CandidateReviewSnapshots(
        raw=[snapshot] if include_snapshots else [],
        filtered=[snapshot] if include_snapshots else [],
        ranked=[snapshot] if include_snapshots else [],
    )
    return LiveEvalEvidence(
        status="pass",
        query="find oxides",
        candidate_snapshots=snapshots,
        real_source_used=real_source_used,
    )


def _tester_report(
    status: str,
    summary: str,
    failed_stage: str | None = None,
    live_evaluation: LiveEvalEvidence | None = None,
) -> RetrievalTesterReport:
    return RetrievalTesterReport(
        status=status,
        failed_stage=failed_stage,
        summary=summary,
        evidence={},
        live_evaluation=live_evaluation,
        recommended_debug_focus=[],
        offline_commands=[],
        live_commands=[],
    )


def _critic_report(verdict: str = "agree", blocked_reason: str | None = None) -> MaterialsQueryCriticReport:
    if verdict == "blocked":
        return MaterialsQueryCriticReport(
            verdict="blocked",
            summary="critic blocked",
            blocked_reason=blocked_reason or "missing evidence",
        )
    if verdict == "disagree":
        return MaterialsQueryCriticReport(
            verdict="disagree",
            summary="Tester conclusion is chemically unsupported.",
            material_findings=["Ranked candidates do not support the claimed material family."],
            owning_modules=["src/matsci_agent/tools/mp_retriever.py"],
            recommended_fix_order=["preserve required chemistry filters"],
            notes_for_debugger=["inspect query compilation"],
        )
    return MaterialsQueryCriticReport(verdict="agree", summary="Tester conclusion is scientifically supported.")


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
        requires_tester_refresh=status in {"accepted", "needs_tester_refresh"},
        tester_refresh_reason=tester_refresh_reason,
        review_notes=["rerun evaluator"] if status in {"accepted", "needs_tester_refresh"} else [],
        acceptance_criteria=[],
    )


def _sequence_runner[T](results: Sequence[T], calls: list[object]):
    values = list(results)

    async def runner(payload):
        calls.append(payload)
        return values[len(calls) - 1]

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


def test_orchestrator_dual_review_passes_only_with_real_evidence(tmp_path: Path):
    tester_calls: list[RetrievalTesterInput] = []
    critic_calls: list[MaterialsQueryCriticInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("pass", "tester passed", live_evaluation=_live_evidence())],
        critic_results=[_critic_report("agree")],
        tester_calls=tester_calls,
        critic_calls=critic_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "pass"
    assert report.stop_reason == "dual_review_pass"
    assert report.attempt_count == 1
    assert report.attempts[0].stop_reason_fragment == "dual_review_pass"
    assert len(tester_calls) == 1
    assert critic_calls[0].objective == "find oxides"
    assert critic_calls[0].review_evidence == report.latest_tester_report.live_evaluation
    artifact_dir = Path(report.artifact_dir or "")
    assert (artifact_dir / "attempts/1/materials_query_critic_input.json").is_file()
    assert (artifact_dir / "attempts/1/materials_query_critic_report.json").is_file()


@pytest.mark.parametrize(
    ("real_source_used", "include_snapshots"),
    [(False, True), (True, False)],
)
def test_orchestrator_blocks_dual_approval_without_real_candidate_evidence(
    tmp_path: Path, real_source_used: bool, include_snapshots: bool
):
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report(
                "pass",
                "tester passed",
                live_evaluation=_live_evidence(real_source_used=real_source_used, include_snapshots=include_snapshots),
            )
        ],
        critic_results=[_critic_report("agree")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "blocked"
    assert report.stop_reason == "scientific_evidence_blocked"


def test_orchestrator_stops_when_tester_is_blocked_without_critic(tmp_path: Path):
    critic_calls: list[MaterialsQueryCriticInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("blocked", "live unavailable")],
        critic_calls=critic_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "blocked"
    assert report.stop_reason == "tester_blocked"
    assert not critic_calls


def test_tester_pass_critic_dissent_enters_repair_loop(tmp_path: Path):
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("pass", "tester passed", live_evaluation=_live_evidence())],
        critic_results=[_critic_report("disagree")],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("fail", "patch not sufficient")],
        debugger_calls=debugger_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "fail"
    assert report.stop_reason == "verifier_fail"
    assert len(debugger_calls) == 1


def test_tester_failure_critic_agreement_enters_repair_loop(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "zero hits", "mp_zero_results")],
        critic_results=[_critic_report("agree")],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("fail", "patch not sufficient")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "fail"
    assert report.stop_reason == "verifier_fail"


def test_repair_loop_uses_configured_branch_prefix(tmp_path: Path):
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "zero hits", "mp_zero_results")],
        critic_results=[_critic_report("agree")],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("fail", "patch not sufficient")],
        debugger_calls=debugger_calls,
    )
    harness.settings = MultiAgentSettings(
        repo_root=harness.settings.repo_root,
        enable_live_mp=True,
        enable_git_write=True,
        worktree_root=harness.settings.worktree_root,
        repair_branch_prefix="retrieval-fix-retry",
    )

    asyncio.run(harness.run("find oxides"))

    assert debugger_calls[0].target_branch_prefix == "retrieval-fix-retry"


def test_tester_failure_critic_dissent_refreshes_before_patch(tmp_path: Path):
    tester_calls: list[RetrievalTesterInput] = []
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "zero hits", "mp_zero_results"),
            _tester_report("pass", "fixed assessment", live_evaluation=_live_evidence()),
        ],
        critic_results=[_critic_report("disagree"), _critic_report("agree")],
        tester_calls=tester_calls,
        debugger_calls=debugger_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.stop_reason == "dual_review_pass"
    assert report.attempt_count == 2
    assert report.attempts[0].stop_reason_fragment == "critic_disagreement_refresh"
    assert tester_calls[1].refresh_feedback == RefreshFeedback(
        source="critic",
        summary="Tester conclusion is chemically unsupported.",
        findings=["Ranked candidates do not support the claimed material family."],
    )
    assert not debugger_calls


def test_verifier_accepted_requires_fresh_tester_and_critic_cycle(tmp_path: Path):
    tester_calls: list[RetrievalTesterInput] = []
    critic_calls: list[MaterialsQueryCriticInput] = []
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "zero hits", "mp_zero_results"),
            _tester_report("pass", "fixed", live_evaluation=_live_evidence()),
        ],
        critic_results=[_critic_report("agree"), _critic_report("agree")],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("accepted", "patch accepted")],
        tester_calls=tester_calls,
        critic_calls=critic_calls,
        debugger_calls=debugger_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.stop_reason == "dual_review_pass"
    assert report.attempt_count == 2
    assert report.attempts[0].stop_reason_fragment == "verifier_accepted_refresh"
    assert tester_calls[1].refresh_feedback is not None
    assert tester_calls[1].refresh_feedback.source == "verifier"
    assert len(critic_calls) == 2
    assert debugger_calls[0].existing_branch_name is None


def test_verifier_refresh_rebinds_agents_to_repair_worktree(tmp_path: Path):
    repair_worktree = tmp_path / "worktrees" / "retrieval-fix-1"
    repair_worktree.mkdir(parents=True)
    rebound_roots: list[Path] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "zero hits", "mp_zero_results"),
            _tester_report("pass", "fixed", live_evaluation=_live_evidence()),
        ],
        critic_results=[_critic_report("agree"), _critic_report("agree")],
        debugger_results=[_debugger_report(worktree_path=str(repair_worktree))],
        verifier_results=[_verifier_report("accepted", "patch accepted")],
    )
    harness.runtime_rebinder = rebound_roots.append

    report = asyncio.run(harness.run("find oxides"))

    assert report.stop_reason == "dual_review_pass"
    assert rebound_roots == [repair_worktree]


def test_live_scenario_repair_uses_exact_constraints_and_retests_repaired_worktree(tmp_path: Path):
    repair_worktree = tmp_path / "worktrees" / "retrieval-fix-1"
    repair_worktree.mkdir(parents=True)
    scenario = get_live_scenario("volume")
    evaluator_settings: list[Path] = []
    tester_calls: list[RetrievalTesterInput] = []
    evidence = [
        LiveEvalEvidence(status="fail", query=scenario.query, failed_stage="mp_zero_results"),
        _live_evidence().model_copy(
            update={
                "query": scenario.query,
                "result_counts": StageCounts(raw_count=1, filtered_count=1, ranked_count=1),
            }
        ),
    ]
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "volume failed", "mp_zero_results"),
            _tester_report("pass", "volume fixed"),
        ],
        critic_results=[_critic_report("agree"), _critic_report("agree")],
        debugger_results=[_debugger_report(worktree_path=str(repair_worktree))],
        verifier_results=[_verifier_report("accepted", "patch accepted")],
        tester_calls=tester_calls,
    )

    def evaluate(settings, _payload):
        evaluator_settings.append(settings.repo_root)
        return evidence.pop(0)

    harness.scenario_evaluator = evaluate
    harness.repair_test_validator = lambda *_args: RepairTestEvidence(status="pass")

    report = asyncio.run(harness.run(scenario.query, scenario=scenario))

    assert report.status == "pass"
    assert tester_calls[0].scenario_name == "volume"
    assert tester_calls[0].live_evaluation_input is not None
    assert tester_calls[0].live_evaluation_input.constraints == scenario.constraints
    assert evaluator_settings == [tmp_path, repair_worktree]
    assert report.latest_repair_test_evidence is not None
    assert report.latest_repair_test_evidence.status == "pass"


def test_live_scenario_repair_blocks_after_failed_deterministic_test_evidence(tmp_path: Path):
    repair_worktree = tmp_path / "worktrees" / "retrieval-fix-1"
    repair_worktree.mkdir(parents=True)
    scenario = get_live_scenario("formation_energy")
    verifier_calls: list[FinalVerifierInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "filter failure", "llm_policy_filter")],
        critic_results=[_critic_report("agree")],
        debugger_results=[_debugger_report(worktree_path=str(repair_worktree))],
        verifier_results=[_verifier_report("accepted", "incorrectly accepted")],
        verifier_calls=verifier_calls,
    )
    harness.scenario_evaluator = lambda _settings, _payload: LiveEvalEvidence(
        status="fail", query=scenario.query, failed_stage="llm_policy_filter"
    )
    harness.repair_test_validator = lambda *_args: RepairTestEvidence(
        status="fail",
        changed_source_files=["src/matsci_agent/tools/policy_filter.py"],
        issues=["changed production-file coverage decreased"],
    )

    report = asyncio.run(harness.run(scenario.query, scenario=scenario))

    assert report.stop_reason == "repair_test_evidence_failed"
    assert verifier_calls[0].repair_test_evidence is not None
    assert verifier_calls[0].repair_test_evidence.status == "fail"


def test_live_scenario_uses_scoped_pass_when_tester_reports_blocked(tmp_path: Path):
    scenario = get_live_scenario("volume")
    tester_calls: list[RetrievalTesterInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("blocked", "agent tool error")],
        critic_results=[_critic_report("agree")],
        tester_calls=tester_calls,
    )
    harness.scenario_evaluator = lambda _settings, _payload: _live_evidence().model_copy(
        update={
            "query": scenario.query,
            "result_counts": StageCounts(raw_count=1, filtered_count=1, ranked_count=1),
        }
    )

    report = asyncio.run(harness.run(scenario.query, scenario=scenario))

    assert report.status == "pass"
    assert report.latest_tester_report is not None
    assert report.latest_tester_report.status == "pass"
    assert tester_calls[0].live_evaluation_input is not None


def test_accepted_patch_on_third_cycle_exhausts_validation_budget(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "fail 1", "mp_zero_results"),
            _tester_report("fail", "fail 2", "mp_zero_results"),
            _tester_report("fail", "fail 3", "mp_zero_results"),
        ],
        critic_results=[_critic_report("agree"), _critic_report("agree"), _critic_report("agree")],
        debugger_results=[_debugger_report(), _debugger_report(), _debugger_report()],
        verifier_results=[
            _verifier_report("accepted", "accepted 1"),
            _verifier_report("accepted", "accepted 2"),
            _verifier_report("accepted", "accepted 3"),
        ],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "fail"
    assert report.stop_reason == "review_cycle_exhausted"
    assert report.attempt_count == 3
    assert report.attempts[-1].stop_reason_fragment == "review_cycle_exhausted"


def test_critic_refresh_on_third_cycle_exhausts_validation_budget(tmp_path: Path):
    debugger_calls: list[CodexDebuggerInput] = []
    harness = _make_harness(
        tmp_path,
        tester_results=[
            _tester_report("fail", "fail 1", "mp_zero_results"),
            _tester_report("fail", "fail 2", "mp_zero_results"),
            _tester_report("fail", "fail 3", "mp_zero_results"),
        ],
        critic_results=[_critic_report("disagree"), _critic_report("disagree"), _critic_report("disagree")],
        debugger_calls=debugger_calls,
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "fail"
    assert report.stop_reason == "review_cycle_exhausted"
    assert report.attempts[-1].stop_reason_fragment == "review_cycle_exhausted"
    assert not debugger_calls


def test_orchestrator_stops_when_critic_is_blocked(tmp_path: Path):
    harness = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report("blocked", "missing evidence")],
    )

    report = asyncio.run(harness.run("find oxides"))

    assert report.status == "blocked"
    assert report.stop_reason == "critic_blocked"
    assert report.next_step == "missing evidence"


def test_orchestrator_stops_when_debugger_or_verifier_cannot_continue(tmp_path: Path):
    debugger_blocked = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report("agree")],
        debugger_results=[_debugger_report(status="blocked", branch_name=None, worktree_path=None)],
    )
    verifier_failed = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report("agree")],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("fail", "patch not sufficient")],
    )
    verifier_blocked = _make_harness(
        tmp_path,
        tester_results=[_tester_report("fail", "fail", "mp_zero_results")],
        critic_results=[_critic_report("agree")],
        debugger_results=[_debugger_report()],
        verifier_results=[_verifier_report("blocked", "cannot inspect patch")],
    )

    blocked_report = asyncio.run(debugger_blocked.run("find oxides"))
    failed_report = asyncio.run(verifier_failed.run("find oxides"))
    verifier_blocked_report = asyncio.run(verifier_blocked.run("find oxides"))

    assert blocked_report.stop_reason == "debugger_blocked"
    assert failed_report.stop_reason == "verifier_fail"
    assert verifier_blocked_report.stop_reason == "verifier_blocked"


def test_critic_contract_normalizes_contradictory_agreement_and_rejects_missing_block_reason():
    normalized_agreement = MaterialsQueryCriticReport(
        verdict="agree",
        summary="looks good",
        notes_for_debugger=["snapshot evidence is internally consistent"],
        informational_notes=["no material findings"],
    )
    assert normalized_agreement.notes_for_debugger == []
    assert normalized_agreement.informational_notes == [
        "no material findings",
        "snapshot evidence is internally consistent",
    ]
    agreement_with_extras = MaterialsQueryCriticReport(
        verdict="agree",
        summary="looks good",
        material_findings=["contradiction"],
        owning_modules=["src/matsci_agent/tools/mp_retriever.py"],
    )
    assert agreement_with_extras.verdict == "agree"
    assert agreement_with_extras.material_findings == []
    assert agreement_with_extras.owning_modules == []
    assert agreement_with_extras.informational_notes == [
        "contradiction",
        "Critic referenced module: src/matsci_agent/tools/mp_retriever.py",
    ]
    with pytest.raises(ValueError, match="requires blocked_reason"):
        MaterialsQueryCriticReport(verdict="blocked", summary="cannot inspect")
    with pytest.raises(ValueError, match="must not include findings"):
        MaterialsQueryCriticReport(
            verdict="blocked",
            summary="cannot inspect",
            material_findings=["unsupported"],
            blocked_reason="missing live evidence",
        )
    with pytest.raises(ValueError, match="must match verifier status"):
        FinalVerifierReport(status="accepted", summary="patch accepted")
