from __future__ import annotations

from multiagent.artifacts import HarnessArtifactStore
from multiagent.schemas import (
    LiveEvalInput,
    LiveEvalScenario,
    LiveEvalScenarioResult,
    LiveEvalSuiteReport,
)
from multiagent.settings import MultiAgentSettings
from multiagent.tools import run_scoped_live_evaluation
from matsci_agent.schemas import DiscoveryConstraints, FloatRange, MPFilters


LIVE_EVAL_SCENARIOS: tuple[LiveEvalScenario, ...] = (
    LiveEvalScenario(
        name="oxide_semiconductor_constraints",
        query="Find oxide semiconductor materials without cobalt and band gap above 2 eV.",
        constraints=DiscoveryConstraints(
            required_elements=["O"],
            banned_elements=["Co"],
            min_band_gap_ev=2.0,
            mp_filters=MPFilters(elements=["O"], exclude_elements=["Co"], is_metal=False),
        ),
        require_target_quality=True,
    ),
    LiveEvalScenario(
        name="lead_free_perovskite_intent",
        query="Find lead-free perovskite materials with band gap above 1 eV.",
        constraints=DiscoveryConstraints(
            banned_elements=["Pb"],
            min_band_gap_ev=1.0,
            mp_filters=MPFilters(exclude_elements=["Pb"]),
        ),
        require_target_quality=True,
    ),
    LiveEvalScenario(
        name="formation_energy",
        query="Find Materials Project entries with formation energy below -1 eV.",
        constraints=DiscoveryConstraints(mp_filters=MPFilters(formation_energy=FloatRange(max=-1.0))),
    ),
    LiveEvalScenario(
        name="energy_above_hull",
        query="Find Materials Project entries with energy above hull below 0.05 eV.",
        constraints=DiscoveryConstraints(
            max_energy_above_hull=0.05,
            mp_filters=MPFilters(energy_above_hull=FloatRange(max=0.05)),
        ),
    ),
    LiveEvalScenario(
        name="density",
        query="Find Materials Project entries with density below 5 g/cm3.",
        constraints=DiscoveryConstraints(mp_filters=MPFilters(density=FloatRange(max=5.0))),
    ),
    LiveEvalScenario(
        name="volume",
        query="Find Materials Project entries with volume below 150 angstrom cubed.",
        constraints=DiscoveryConstraints(mp_filters=MPFilters(volume=FloatRange(max=150.0))),
    ),
    LiveEvalScenario(
        name="has_props",
        query="Find Materials Project entries with dielectric properties.",
        constraints=DiscoveryConstraints(mp_filters=MPFilters(has_props=["dielectric"])),
    ),
    LiveEvalScenario(
        name="cubic_symmetry",
        query="Find cubic Materials Project entries.",
        constraints=DiscoveryConstraints(mp_filters=MPFilters(crystal_system="cubic")),
    ),
)


def get_live_scenario(name: str) -> LiveEvalScenario:
    for scenario in LIVE_EVAL_SCENARIOS:
        if scenario.name == name:
            return scenario
    allowed = ", ".join(scenario.name for scenario in LIVE_EVAL_SCENARIOS)
    raise ValueError(f"unknown live scenario: {name}. Allowed values: {allowed}")


def scenario_assertion_failures(scenario: LiveEvalScenario, evidence) -> list[str]:
    failures: list[str] = []
    if evidence.status == "pass" and not evidence.real_source_used:
        failures.append("real Materials Project source was not used")
    if evidence.status == "pass" and evidence.result_counts.ranked_count < scenario.min_ranked_count:
        failures.append(f"ranked count below minimum {scenario.min_ranked_count}")
    if evidence.status == "pass" and scenario.require_target_quality and evidence.result_counts.search_space_target_count == 0:
        failures.append("Search Space Expansion returned no targets")
    return failures


def run_live_suite(
    settings: MultiAgentSettings,
    *,
    evaluator=None,
) -> LiveEvalSuiteReport:
    store = HarnessArtifactStore.create(settings, "multiagent_live_eval_suite")
    results: list[LiveEvalScenarioResult] = []

    for scenario in LIVE_EVAL_SCENARIOS:
        payload = LiveEvalInput(
            query=scenario.query,
            constraints=scenario.constraints,
            allow_live_mp=settings.enable_live_mp,
        )
        evidence = evaluator.evaluate(payload) if evaluator is not None else run_scoped_live_evaluation(settings, payload)
        result = LiveEvalScenarioResult(
            scenario=scenario,
            evidence=evidence,
            assertion_failures=scenario_assertion_failures(scenario, evidence),
        )
        results.append(result)
        store.write_model(f"scenarios/{scenario.name}.json", result)

    if any(result.evidence.status == "blocked" for result in results):
        status = "blocked"
    elif any(result.evidence.status != "pass" or result.assertion_failures for result in results):
        status = "fail"
    else:
        status = "pass"
    report = LiveEvalSuiteReport(status=status, artifact_dir=str(store.run_dir), scenarios=results)
    store.write_model("live_suite_report.json", report)
    return report
