from matsci_agent.agents.planner import ChemistryIntentAgent
from matsci_agent.guardrails.capability import CapabilityGuardrail
from matsci_agent.schemas import DiscoveryConstraints, ParsedDiscoveryIntent


def test_planner_builds_band_gap_screening_plan_with_selective_recalc():
    agent = ChemistryIntentAgent(
        parser_fn=lambda _goal: ParsedDiscoveryIntent(
            requested_material_class="semiconductor",
            constraints=DiscoveryConstraints(
                banned_elements=["Si"],
                min_band_gap_ev=2.0,
                calculate_matgl=True,
            ),
        )
    )
    plan = agent.plan(
        "Find semiconductor materials without silicon with band gap over 2 eV and redo calculation of top 1 candidate.",
        DiscoveryConstraints(),
        explicit_base_fields=set(),
    )

    assert plan.task_class == "band_gap_screening"
    assert plan.source_universe == "materials_project_entries"
    assert plan.requested_material_class == "semiconductor"
    assert plan.parsed_constraints.banned_elements == ["Si"]
    assert plan.parsed_constraints.min_band_gap_ev == 2.0
    assert plan.execution_policy.calculate_matgl is True
    assert plan.execution_policy.recalculate_top_n == 1


def test_planner_extracts_selective_recalc_when_prompt_uses_the_top_n_phrase():
    agent = ChemistryIntentAgent(
        parser_fn=lambda _goal: ParsedDiscoveryIntent(
            requested_material_class="semiconductor",
            constraints=DiscoveryConstraints(calculate_matgl=True),
        )
    )

    plan = agent.plan(
        "Find semiconductor materials and recalculate the top 2 candidates with MatGL.",
        DiscoveryConstraints(),
        explicit_base_fields=set(),
    )

    assert plan.task_class == "band_gap_screening"
    assert plan.source_universe == "materials_project_entries"
    assert plan.requested_material_class == "semiconductor"
    assert plan.execution_policy.calculate_matgl is True
    assert plan.execution_policy.recalculate_top_n == 2


def test_guardrail_refuses_diffusivity_task():
    agent = ChemistryIntentAgent(
        parser_fn=lambda _goal: ParsedDiscoveryIntent(requested_material_class="unknown")
    )
    plan = agent.plan(
        "Calculate diffusivity in bulk materials with long MD trajectories.",
        DiscoveryConstraints(),
        explicit_base_fields=set(),
    )

    assessment = CapabilityGuardrail().assess(plan)

    assert plan.task_class == "diffusivity_simulation"
    assert plan.source_universe == "materials_project_entries"
    assert plan.requested_material_class == "unknown"
    assert assessment.supported is False
    assert assessment.reason_code == "unsupported_diffusivity"
    assert "Diffusivity is unsupported" in assessment.reason_message


def test_planner_supports_generic_mp_property_screening():
    agent = ChemistryIntentAgent(
        parser_fn=lambda _goal: ParsedDiscoveryIntent(
            requested_material_class="perovskite",
            constraints=DiscoveryConstraints(),
        )
    )
    plan = agent.plan(
        "Find lead-free perovskite materials with formation energy below -1 eV.",
        DiscoveryConstraints(),
        explicit_base_fields=set(),
    )

    assessment = CapabilityGuardrail().assess(plan)

    assert plan.task_class == "mp_property_screening"
    assert assessment.supported is True
