import pytest

from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    DiscoveryPlan,
    ExecutionPolicy,
    PolicyFilterInput,
)
from matsci_agent.tools.policy_filter import PolicyFilter, PolicyFilterError


def _plan(application_intent: str, practicality_mode: str) -> DiscoveryPlan:
    return DiscoveryPlan(
        research_goal_raw="Find semiconductor bulk materials",
        task_class="band_gap_screening",
        parsed_constraints=DiscoveryConstraints(top_k=5),
        application_intent=application_intent,
        material_class="bulk_inorganic",
        practicality_mode=practicality_mode,
        execution_policy=ExecutionPolicy(),
    )


def test_policy_filter_valid_batch_response_sets_candidate_provenance():
    tool = PolicyFilter(
        inference_fn=lambda _payload: {
            "policy_name": "practical_screening",
            "decisions": [
                {"material_id": "c1", "keep": False, "reasons": ["radioactive fluoride not practical semiconductor"]},
                {"material_id": "c2", "keep": True, "reasons": []},
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[
            Candidate(material_id="c1", formula="AcF3", features={"elements": ["Ac", "F"]}),
            Candidate(material_id="c2", formula="AlN", features={"elements": ["Al", "N"]}),
        ],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    out = tool.run(payload)

    assert [c.material_id for c in out.filtered_candidates] == ["c2"]
    rejected = {r.candidate.material_id: r for r in out.records}
    assert rejected["c1"].passed is False
    assert rejected["c1"].candidate.features["filter_source"] == "llm"
    assert out.provenance.output_summary["provider"] == "auto"


def test_policy_filter_practical_request_can_drop_sulfate_like_salt():
    tool = PolicyFilter(
        inference_fn=lambda _payload: {
            "policy_name": "practical_screening",
            "decisions": [
                {"material_id": "c1", "keep": False, "reasons": ["molecular salt not practical semiconductor"]},
                {"material_id": "c2", "keep": True, "reasons": []},
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[
            Candidate(material_id="c1", formula="Al2(SO4)3", features={"elements": ["Al", "S", "O"]}),
            Candidate(material_id="c2", formula="AlN", features={"elements": ["Al", "N"]}),
        ],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    out = tool.run(payload)

    assert [c.material_id for c in out.filtered_candidates] == ["c2"]


def test_policy_filter_invalid_json_fails_closed():
    tool = PolicyFilter(inference_fn=lambda _payload: "not-json")
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="AcF3", features={"elements": ["Ac", "F"]})],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    with pytest.raises(PolicyFilterError) as exc_info:
        tool.run(payload)

    assert exc_info.value.code == "policy_filter_invalid_json"


def test_policy_filter_missing_candidate_decision_fails_closed():
    tool = PolicyFilter(
        inference_fn=lambda _payload: {
            "policy_name": "practical_screening",
            "decisions": [
                {"material_id": "c1", "keep": True, "reasons": []},
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[
            Candidate(material_id="c1", formula="AlN", features={"elements": ["Al", "N"]}),
            Candidate(material_id="c2", formula="GaN", features={"elements": ["Ga", "N"]}),
        ],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    with pytest.raises(PolicyFilterError) as exc_info:
        tool.run(payload)

    assert exc_info.value.code == "policy_filter_missing_candidate_decisions"


def test_policy_filter_reason_limits_enforced():
    tool = PolicyFilter(
        inference_fn=lambda _payload: {
            "policy_name": "practical_screening",
            "decisions": [
                {
                    "material_id": "c1",
                    "keep": False,
                    "reasons": ["x" * 81],
                }
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="Al2(SO4)3", features={"elements": ["Al", "S", "O"]})],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    with pytest.raises(PolicyFilterError) as exc_info:
        tool.run(payload)

    assert exc_info.value.code == "policy_filter_reason_too_long"


def test_policy_filter_without_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tool = PolicyFilter()
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="AlN", features={"elements": ["Al", "N"]})],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    with pytest.raises(PolicyFilterError) as exc_info:
        tool.run(payload)

    assert exc_info.value.code == "policy_filter_llm_request_failed"
