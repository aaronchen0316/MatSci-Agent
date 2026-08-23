from __future__ import annotations

import subprocess
from pathlib import Path

import multiagent.audit as audit
from multiagent.schemas import (
    CodexDebuggerReport,
    FinalVerifierReport,
    HarnessRunReport,
    LiveEvalEvidence,
    RepairTestEvidence,
    RetrievalTesterReport,
)
from multiagent.settings import MultiAgentSettings


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, artifact_root=tmp_path / "artifacts")


def _passing_report(*, commit_sha: str) -> HarnessRunReport:
    return HarnessRunReport(
        status="pass",
        summary="accepted",
        next_step="publish",
        stop_reason="dual_review_pass",
        attempt_count=1,
        branch_name="fix/volume",
        latest_tester_report=RetrievalTesterReport(
            status="pass",
            summary="live pass",
            live_evaluation=LiveEvalEvidence(status="pass", query="volume", real_source_used=True),
        ),
        latest_debugger_report=CodexDebuggerReport(
            status="patched",
            branch_name="fix/volume",
            commit_sha=commit_sha,
            change_summary="repair",
        ),
        latest_repair_test_evidence=RepairTestEvidence(status="pass"),
        latest_verifier_report=FinalVerifierReport(
            status="accepted",
            summary="accepted",
            requires_tester_refresh=True,
        ),
    )


def test_audit_rejects_stale_debugger_evidence_sha(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)

    def fake_run(args, _cwd):
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return subprocess.CompletedProcess(args, 0, "head-sha\n", "")
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["git", "diff", "--name-only"]:
            return subprocess.CompletedProcess(args, 0, "src/matsci_agent/tools/mp_retriever.py\ntests/test_mp_retriever.py\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(audit, "_run", fake_run)
    monkeypatch.setattr(audit, "_matching_artifact", lambda *_args: (tmp_path, _passing_report(commit_sha="old-sha")))
    monkeypatch.setattr(audit, "_load_production_artifact", lambda *_args, **_kwargs: "harness debugger commit SHA does not match repair branch head")

    record = audit.audit_repair_branch(settings, "fix/volume")

    assert record.status == "rejected"
    assert record.safe_namespace
    assert record.descends_from_target_main
    assert record.product_only_diff
    assert not record.evidence_matches_head
    assert record.debugger_committed
    assert record.changed_tests_proven
    assert record.verifier_accepted
    assert record.fresh_live_passed
    assert "debugger commit SHA does not match branch head" in " ".join(record.reasons)


def test_audit_rejects_legacy_non_product_branch_without_git_writes(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    commands: list[list[str]] = []

    def fake_run(args, _cwd):
        commands.append(args)
        if args[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(args, 0, "legacy-sha\n", "")
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        if args[:3] == ["git", "diff", "--name-only"]:
            return subprocess.CompletedProcess(args, 0, "src/multiagent/tools.py\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(audit, "_run", fake_run)
    monkeypatch.setattr(audit, "_matching_artifact", lambda *_args: (None, None))

    record = audit.audit_repair_branch(settings, "retrieval-fix-retry")

    assert record.status == "rejected"
    assert not record.safe_namespace
    assert not record.descends_from_target_main
    assert not record.product_only_diff
    assert any("fix/<issue>" in reason for reason in record.reasons)
    assert any("non-product" in reason for reason in record.reasons)
    assert not any(command[:2] in (["git", "push"], ["git", "worktree"]) for command in commands)
