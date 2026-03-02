from matsci_agent.schemas import Candidate, PropertyPredictorInput
from matsci_agent.tools.property_predictor import PropertyPredictor


def test_predictor_prefers_mp_band_gap_when_available_and_not_forced():
    predictor = PropertyPredictor()
    payload = PropertyPredictorInput(
        candidates=[
            Candidate(
                material_id="c1",
                formula="AlN",
                features={"mp_band_gap_ev": 5.7, "nsites": 8},
            )
        ],
        goal="find semiconductors",
        calculate_matgl=False,
    )

    out = predictor.run(payload)

    assert out.predictions[0].predicted.backend == "materials_project_band_gap"
    assert out.predictions[0].predicted.band_gap_ev == 5.7
    assert out.provenance.output_summary["used_mp_count"] == 1
    assert out.predictions[0].candidate.features["band_gap_source"] == "materials_project"
    assert out.predictions[0].candidate.features["matgl_forced"] is False


def test_predictor_forced_matgl_applies_deterministic_limits_and_provenance_tags():
    predictor = PropertyPredictor()
    payload = PropertyPredictorInput(
        candidates=[
            Candidate(
                material_id="mp-20",
                formula="AlN",
                features={"mp_band_gap_ev": 5.7, "nsites": 8},
            ),
            Candidate(
                material_id="mp-3",
                formula="Fe2VAl",
                features={"mp_band_gap_ev": None, "nsites": 12},
            ),
            Candidate(
                material_id="mp-1",
                formula="Ni3Al",
                features={"mp_band_gap_ev": None, "nsites": 55},
            ),
        ],
        goal="force recalculate with matgl",
        calculate_matgl=True,
        matgl_max_recalc_entries=1,
        matgl_max_atoms=50,
    )

    out = predictor.run(payload)

    assert out.provenance.output_summary["forced_matgl_count"] == 1
    assert out.provenance.output_summary["matgl_skipped_count"] == 2
    assert out.provenance.output_summary["matgl_selected_material_ids"] == ["mp-3"]

    by_id = {p.candidate.material_id: p for p in out.predictions}
    assert all(p.candidate.features["matgl_forced"] is True for p in out.predictions)
    assert by_id["mp-20"].candidate.features["matgl_skipped_reason"] == "recalc_limit_reached"
    assert by_id["mp-20"].candidate.features["band_gap_source"] == "materials_project"
    assert by_id["mp-1"].candidate.features["matgl_skipped_reason"] == "atoms_too_high_or_missing_nsites"
    assert by_id["mp-1"].candidate.features["band_gap_source"] == "fallback"
