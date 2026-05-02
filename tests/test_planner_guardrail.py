from matsci_agent.agents.planner import ChemistryIntentAgent
from matsci_agent.guardrails.capability import CapabilityGuardrail
from matsci_agent.schemas import DiscoveryConstraints


def test_planner_builds_band_gap_screening_plan_with_selective_recalc():
    agent = ChemistryIntentAgent(
        parser_fn=lambda _goal: DiscoveryConstraints(
            banned_elements=["Si"],
            min_band_gap_ev=2.0,
            calculate_matgl=True,
        )
    )
    plan = agent.plan(
        "Find semiconductor materials without silicon with band gap over 2 eV and redo calculation of top 1 candidate.",
        DiscoveryConstraints(),
        explicit_base_fields=set(),
    )

    assert plan.task_class == "band_gap_screening"
    assert plan.parsed_constraints.banned_elements == ["Si"]
    assert plan.parsed_constraints.min_band_gap_ev == 2.0
    assert plan.execution_policy.calculate_matgl is True
    assert plan.execution_policy.recalculate_top_n == 1


def test_guardrail_refuses_diffusivity_task():
    agent = ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints())
    plan = agent.plan(
        "Calculate diffusivity in bulk materials with long MD trajectories.",
        DiscoveryConstraints(),
        explicit_base_fields=set(),
    )

    assessment = CapabilityGuardrail().assess(plan)

    assert plan.task_class == "diffusivity_simulation"
    assert assessment.supported is False
    assert assessment.reason_code == "unsupported_diffusivity"
    assert "Diffusivity is unsupported" in assessment.reason_message
