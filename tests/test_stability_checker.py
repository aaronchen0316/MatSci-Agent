from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    PredictedProperties,
    PropertyPredictionRecord,
    StabilityCheckerInput,
)
from matsci_agent.tools.stability_checker import StabilityChecker


def test_stability_checker_prefers_mp_energy_above_hull():
    tool = StabilityChecker()
    payload = StabilityCheckerInput(
        predictions=[
            PropertyPredictionRecord(
                candidate=Candidate(
                    material_id="c1",
                    formula="AlN",
                    features={"mp_energy_above_hull": 0.012, "nsites": 8},
                ),
                predicted=PredictedProperties(band_gap_ev=5.7, uncertainty=0.2, backend="mp"),
            )
        ],
        constraints=DiscoveryConstraints(max_energy_above_hull=0.05),
    )

    out = tool.run(payload)

    record = out.records[0]
    assert record.stability.energy_above_hull == 0.012
    assert record.stability.source == "materials_project"
    assert record.stability.method == "materials_project_energy_above_hull"
    assert record.stability.used_relaxation is False


def test_stability_checker_missing_hull_returns_unknown():
    tool = StabilityChecker()
    payload = StabilityCheckerInput(
        predictions=[
            PropertyPredictionRecord(
                candidate=Candidate(
                    material_id="c2",
                    formula="Fe2VAl",
                    features={"mp_energy_above_hull": None, "nsites": 12},
                ),
                predicted=PredictedProperties(band_gap_ev=1.3, uncertainty=1.0, backend="matgl"),
            )
        ],
        constraints=DiscoveryConstraints(max_energy_above_hull=0.2),
    )

    out = tool.run(payload)

    record = out.records[0]
    assert record.stability.energy_above_hull is None
    assert record.stability.is_stable is None
    assert record.stability.source == "unknown"
    assert record.stability.method == "stability_unknown_no_mp_hull"
    assert record.stability.used_relaxation is False
