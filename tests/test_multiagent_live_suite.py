from __future__ import annotations

from matsci_agent.multiagent.live_suite import LIVE_EVAL_SCENARIOS, run_live_suite
from matsci_agent.multiagent.schemas import LiveEvalEvidence, StageCounts
from matsci_agent.multiagent.settings import MultiAgentSettings


class PassingEvaluator:
    def evaluate(self, payload):
        return LiveEvalEvidence(
            status="pass",
            query=payload.query,
            result_counts=StageCounts(
                raw_count=2,
                filtered_count=1,
                ranked_count=1,
                search_space_target_count=2,
            ),
            real_source_used=True,
        )


class BlockedEvaluator:
    def evaluate(self, payload):
        return LiveEvalEvidence(status="blocked", query=payload.query, blocked_reason="missing MP_API_KEY")


def _settings(tmp_path) -> MultiAgentSettings:
    return MultiAgentSettings(
        repo_root=tmp_path,
        enable_live_mp=True,
        artifact_root=tmp_path / "artifacts" / "multiagent-runs",
    )


def test_live_suite_defines_eight_required_scenarios():
    assert len(LIVE_EVAL_SCENARIOS) == 8
    assert {scenario.name for scenario in LIVE_EVAL_SCENARIOS} == {
        "oxide_semiconductor_constraints",
        "lead_free_perovskite_intent",
        "formation_energy",
        "energy_above_hull",
        "density",
        "volume",
        "has_props",
        "cubic_symmetry",
    }


def test_live_suite_writes_artifacts_for_passing_scenarios(tmp_path):
    report = run_live_suite(_settings(tmp_path), evaluator=PassingEvaluator())

    assert report.status == "pass"
    assert report.artifact_dir is not None
    artifact_dir = tmp_path / "artifacts" / "multiagent-runs"
    assert len(list(artifact_dir.glob("*/live_suite_report.json"))) == 1
    assert len(list(artifact_dir.glob("*/scenarios/*.json"))) == 8


def test_live_suite_propagates_blocked_evidence(tmp_path):
    report = run_live_suite(_settings(tmp_path), evaluator=BlockedEvaluator())

    assert report.status == "blocked"
