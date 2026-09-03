from types import SimpleNamespace

import pytest

import matsci_agent.agents.search_space_expander as search_space_expander
from matsci_agent.agents.search_space_expander import SearchSpaceExpansionAgent, SearchSpaceExpansionError
from matsci_agent.nlp.parser import FALLBACK_LLM_MODEL, PRIMARY_LLM_MODEL
from matsci_agent.schemas import DiscoveryConstraints, DiscoveryPlan, SearchSpaceExpansionInput


def _plan() -> DiscoveryPlan:
    return DiscoveryPlan(
        research_goal_raw="Find lead-free perovskite materials.",
        task_class="mp_property_screening",
        parsed_constraints=DiscoveryConstraints(banned_elements=["Pb"]),
        requested_material_class="perovskite",
    )


def _response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_expander_normalizes_and_filters_formula_targets():
    agent = SearchSpaceExpansionAgent(
        inference_fn=lambda _payload: {
            "formula_targets": [
                {"formula": "CsSnI₃", "chemsys": "Cs-I-Sn", "confidence": 0.9, "rationale": "tin halide"},
                {"formula": "CsPbI3", "chemsys": "Cs-I-Pb", "confidence": 0.9, "rationale": "contains lead"},
                {"formula": "not-a-formula", "chemsys": "X", "confidence": 0.4, "rationale": "bad"},
                {"formula": "CsSnI3", "chemsys": "Cs-I-Sn", "confidence": 0.8, "rationale": "duplicate"},
            ]
        }
    )
    plan = _plan()

    out = agent.expand(
        SearchSpaceExpansionInput(
            research_goal=plan.research_goal_raw,
            discovery_plan=plan,
            target_count=10,
        )
    )

    assert [target.normalized_formula for target in out.targets] == ["CsSnI3"]
    assert out.targets[0].chemsys == "Cs-I-Sn"


def test_expander_treats_llm_chemsys_as_advisory():
    agent = SearchSpaceExpansionAgent(
        inference_fn=lambda _payload: {
            "formula_targets": [
                {
                    "formula": "Rb2AgBiBr6",
                    "chemsys": "Ag-Br-Br-Rb-Bi",
                    "confidence": 0.77,
                    "rationale": "duplicate bromine in advisory chemsys",
                },
            ]
        }
    )
    plan = _plan()

    out = agent.expand(SearchSpaceExpansionInput(research_goal=plan.research_goal_raw, discovery_plan=plan))

    assert out.targets[0].normalized_formula == "Rb2AgBiBr6"
    assert out.targets[0].chemsys == "Ag-Bi-Br-Rb"


def test_expander_retries_when_response_missing_formula_targets():
    responses = iter(
        [
            {},
            {
                "formula_targets": [
                    {"formula": "CsSnI3", "chemsys": "Cs-I-Sn", "confidence": 0.9},
                ]
            },
        ]
    )
    seen_payloads = []

    def _infer(payload):
        seen_payloads.append(payload)
        return next(responses)

    agent = SearchSpaceExpansionAgent(inference_fn=_infer)
    plan = _plan()

    out = agent.expand(SearchSpaceExpansionInput(research_goal=plan.research_goal_raw, discovery_plan=plan))

    assert [target.normalized_formula for target in out.targets] == ["CsSnI3"]
    assert len(seen_payloads) == 2
    assert "previous_error" in seen_payloads[1]


def test_expander_fails_closed_when_no_valid_targets():
    agent = SearchSpaceExpansionAgent(
        inference_fn=lambda _payload: {
            "formula_targets": [
                {"formula": "CsPbI3", "chemsys": "Cs-I-Pb", "confidence": 0.9},
            ]
        }
    )
    plan = _plan()

    try:
        agent.expand(SearchSpaceExpansionInput(research_goal=plan.research_goal_raw, discovery_plan=plan))
    except SearchSpaceExpansionError as exc:
        assert exc.code == "search_space_expansion_empty"
    else:
        raise AssertionError("expected SearchSpaceExpansionError")


def test_expander_uses_shared_primary_model_and_falls_back_when_unavailable(monkeypatch):
    import openai

    requests: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            assert kwargs["base_url"] == "https://proxy.example/v1"
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            requests.append(kwargs)
            if kwargs["model"] == PRIMARY_LLM_MODEL:
                raise RuntimeError("model not found")
            return _response('{"formula_targets":[{"formula":"CsSnI3"}]}')

        def close(self) -> None:
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY_RAG", "test-key")
    monkeypatch.setenv("MATSCI_LLM_BASE_URL", "https://proxy.example/v1")
    monkeypatch.delenv("MATSCI_NLP_MODEL", raising=False)
    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    agent = SearchSpaceExpansionAgent()
    out = agent.expand(SearchSpaceExpansionInput(research_goal=_plan().research_goal_raw, discovery_plan=_plan()))

    assert [request["model"] for request in requests] == [PRIMARY_LLM_MODEL, FALLBACK_LLM_MODEL]
    assert out.provenance.output_summary["model"] == FALLBACK_LLM_MODEL
    assert out.provenance.output_summary["request_attempt_count"] == 2


def test_expander_retries_transient_timeout_with_backoff(monkeypatch):
    import openai

    calls = 0
    sleeps: list[float] = []

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("request timed out")
            return _response('{"formula_targets":[{"formula":"CsSnI3"}]}')

        def close(self) -> None:
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY_RAG", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(search_space_expander.time, "sleep", sleeps.append)

    agent = SearchSpaceExpansionAgent()
    out = agent.expand(SearchSpaceExpansionInput(research_goal=_plan().research_goal_raw, discovery_plan=_plan()))

    assert calls == 2
    assert sleeps == [1.0]
    assert out.provenance.output_summary["request_attempt_count"] == 2


def test_expander_does_not_retry_invalid_json_response(monkeypatch):
    import openai

    calls = 0

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            return _response("not-json")

        def close(self) -> None:
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY_RAG", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    agent = SearchSpaceExpansionAgent()
    with pytest.raises(SearchSpaceExpansionError) as exc_info:
        agent.expand(SearchSpaceExpansionInput(research_goal=_plan().research_goal_raw, discovery_plan=_plan()))

    assert exc_info.value.code == "search_space_expansion_invalid_json"
    assert calls == 1
