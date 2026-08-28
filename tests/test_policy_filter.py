from types import SimpleNamespace

import pytest

import matsci_agent.tools.policy_filter as policy_filter_module
from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    DiscoveryPlan,
    ExecutionPolicy,
    PolicyFilterInput,
)
from matsci_agent.tools.policy_filter import PolicyFilter, PolicyFilterError


def _plan(application_intent: str, practicality_mode: str, task_class: str = "band_gap_screening") -> DiscoveryPlan:
    return DiscoveryPlan(
        research_goal_raw="Find semiconductor bulk materials",
        task_class=task_class,
        parsed_constraints=DiscoveryConstraints(top_k=5),
        application_intent=application_intent,
        source_universe="materials_project_entries",
        requested_material_class="semiconductor",
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
    assert out.provenance.output_summary["provider"] == "openrouter"


def test_policy_filter_runs_for_generic_mp_property_screening():
    tool = PolicyFilter(
        inference_fn=lambda payload: {
            "policy_name": "chemistry_screening",
            "decisions": [
                {"material_id": candidate["material_id"], "keep": True, "reasons": []}
                for candidate in payload["candidates"]
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="CsSnI3", features={"elements": ["Cs", "Sn", "I"]})],
        discovery_plan=_plan("unknown", "unknown", task_class="mp_property_screening"),
    )

    out = tool.run(payload)

    assert [candidate.material_id for candidate in out.filtered_candidates] == ["c1"]


def test_policy_filter_payload_uses_source_universe_and_requested_material_class():
    tool = PolicyFilter(inference_fn=lambda _payload: {"policy_name": "chemistry_screening", "decisions": []})
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="SiC", features={"elements": ["Si", "C"]})],
        discovery_plan=_plan("unknown", "unknown"),
    )

    prompt_payload = tool._prompt_payload(payload)

    assert prompt_payload["discovery_context"]["source_universe"] == "materials_project_entries"
    assert prompt_payload["discovery_context"]["requested_material_class"] == "semiconductor"
    assert "material_class" not in prompt_payload["discovery_context"]


def test_policy_filter_payload_includes_generic_mp_summary_properties():
    tool = PolicyFilter(inference_fn=lambda _payload: {"policy_name": "chemistry_screening", "decisions": []})
    payload = PolicyFilterInput(
        candidates=[
            Candidate(
                material_id="c1",
                formula="SrTiO3",
                features={
                    "elements": ["Sr", "Ti", "O"],
                    "formation_energy": -3.55,
                    "density": 5.1,
                    "volume": 119.6,
                    "efermi": 2.3,
                    "total_magnetization": 0.0,
                },
            )
        ],
        discovery_plan=_plan("unknown", "unknown", task_class="mp_property_screening"),
    )

    candidate = tool._prompt_payload(payload)["candidates"][0]

    assert candidate["formation_energy"] == -3.55
    assert candidate["density"] == 5.1
    assert candidate["volume"] == 119.6
    assert candidate["efermi"] == 2.3
    assert candidate["total_magnetization"] == 0.0


def test_policy_filter_prompt_references_materials_project_entries():
    prompt = PolicyFilter._system_prompt()

    assert "candidate Materials Project entries" in prompt
    assert "preexisting material class" in prompt
    assert "research_goal is authoritative" in prompt


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


def test_policy_filter_reason_text_is_normalized():
    tool = PolicyFilter(
        inference_fn=lambda _payload: {
            "policy_name": "practical_screening",
            "decisions": [
                {
                    "material_id": "c1",
                    "keep": True,
                    "reasons": ["long    reason " + ("x" * 200)],
                }
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="Al2(SO4)3", features={"elements": ["Al", "S", "O"]})],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    out = tool.run(payload)

    reason = out.records[0].reasons[0]
    assert len(reason) <= 160
    assert "  " not in reason


def test_policy_filter_caps_reason_count():
    tool = PolicyFilter(
        inference_fn=lambda _payload: {
            "policy_name": "practical_screening",
            "decisions": [
                {
                    "material_id": "c1",
                    "keep": True,
                    "reasons": ["one", "two", "three", "four"],
                }
            ],
        }
    )
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="AlN", features={"elements": ["Al", "N"]})],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    out = tool.run(payload)

    assert out.records[0].reasons == ["one", "two", "three"]


def test_policy_filter_without_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY_RAG", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MATSCI_LLM_API_KEY_ENV", raising=False)
    tool = PolicyFilter()
    payload = PolicyFilterInput(
        candidates=[
            Candidate(material_id="c1", formula="AcF3", features={"elements": ["Ac", "F"], "nsites": 16}),
            Candidate(material_id="c2", formula="AlN", features={"elements": ["Al", "N"], "nsites": 8}),
        ],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    with pytest.raises(PolicyFilterError) as exc_info:
        tool.run(payload)

    assert exc_info.value.code == "policy_filter_llm_request_failed"


def test_policy_filter_remote_failure_fails_closed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_RAG", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MATSCI_LLM_API_KEY_ENV", raising=False)
    tool = PolicyFilter(provider="openrouter")
    payload = PolicyFilterInput(
        candidates=[
            Candidate(material_id="c1", formula="AcF3", features={"elements": ["Ac", "F"], "nsites": 16}),
            Candidate(material_id="c2", formula="AlN", features={"elements": ["Al", "N"], "nsites": 8}),
        ],
        discovery_plan=_plan("practical_screening", "applied"),
    )

    monkeypatch.setattr(
        tool,
        "_call_openrouter",
        lambda _payload: (_ for _ in ()).throw(
            PolicyFilterError("policy_filter_invalid_json", "bad json")
        ),
    )

    with pytest.raises(PolicyFilterError) as exc_info:
        tool.run(payload)

    assert exc_info.value.code == "policy_filter_invalid_json"


def test_policy_filter_retries_transient_timeout_with_backoff(monkeypatch):
    tool = PolicyFilter(provider="openrouter")
    payload = PolicyFilterInput(
        candidates=[Candidate(material_id="c1", formula="AlN", features={"elements": ["Al", "N"]})],
        discovery_plan=_plan("unknown", "unknown", task_class="mp_property_screening"),
    )
    calls = 0
    sleeps: list[float] = []

    def request(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("request timed out")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"decisions":[{"material_id":"c1","keep":true,"reasons":[]}]}'))]
        )

    monkeypatch.setenv("OPENROUTER_API_KEY_RAG", "test-key")
    monkeypatch.setattr(tool, "_request_openrouter_completion", request)
    monkeypatch.setattr(policy_filter_module.time, "sleep", sleeps.append)

    out = tool.run(payload)

    assert [candidate.material_id for candidate in out.filtered_candidates] == ["c1"]
    assert calls == 2
    assert sleeps == [1.0]
    assert out.provenance.output_summary["request_attempt_count"] == 2
