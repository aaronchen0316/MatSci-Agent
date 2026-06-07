from matsci_agent.agents.search_space_expander import SearchSpaceExpansionAgent, SearchSpaceExpansionError
from matsci_agent.schemas import DiscoveryConstraints, DiscoveryPlan, SearchSpaceExpansionInput


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
    plan = DiscoveryPlan(
        research_goal_raw="Find lead-free perovskite materials.",
        task_class="mp_property_screening",
        parsed_constraints=DiscoveryConstraints(banned_elements=["Pb"]),
        requested_material_class="perovskite",
    )

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
    plan = DiscoveryPlan(
        research_goal_raw="Find lead-free perovskite materials.",
        task_class="mp_property_screening",
        parsed_constraints=DiscoveryConstraints(banned_elements=["Pb"]),
        requested_material_class="perovskite",
    )

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
    plan = DiscoveryPlan(
        research_goal_raw="Find lead-free perovskite materials.",
        task_class="mp_property_screening",
        parsed_constraints=DiscoveryConstraints(banned_elements=["Pb"]),
        requested_material_class="perovskite",
    )

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
    plan = DiscoveryPlan(
        research_goal_raw="Find lead-free perovskite materials.",
        task_class="mp_property_screening",
        parsed_constraints=DiscoveryConstraints(banned_elements=["Pb"]),
        requested_material_class="perovskite",
    )

    try:
        agent.expand(SearchSpaceExpansionInput(research_goal=plan.research_goal_raw, discovery_plan=plan))
    except SearchSpaceExpansionError as exc:
        assert exc.code == "search_space_expansion_empty"
    else:
        raise AssertionError("expected SearchSpaceExpansionError")
