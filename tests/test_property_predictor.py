from matsci_agent.schemas import Candidate, PropertyPredictorInput
from matsci_agent.tools import property_predictor as predictor_module
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


def test_predictor_selective_recalc_limits_forced_matgl_to_top_n():
    predictor = PropertyPredictor()
    payload = PropertyPredictorInput(
        candidates=[
            Candidate(material_id="mp-1", formula="AlN", features={"mp_band_gap_ev": 5.7, "nsites": 8}),
            Candidate(material_id="mp-2", formula="GaN", features={"mp_band_gap_ev": 3.2, "nsites": 8}),
            Candidate(material_id="mp-3", formula="ZnO", features={"mp_band_gap_ev": 2.1, "nsites": 8}),
        ],
        goal="recalculate top 1 candidate",
        calculate_matgl=True,
        recalculate_top_n=1,
        matgl_max_recalc_entries=3,
        matgl_max_atoms=50,
    )

    out = predictor.run(payload)

    assert out.provenance.output_summary["forced_matgl_count"] == 1
    assert out.provenance.output_summary["matgl_selected_material_ids"] == ["mp-1"]


def test_predictor_selective_recalc_falls_back_for_unselected_missing_mp_gap():
    predictor = PropertyPredictor()
    payload = PropertyPredictorInput(
        candidates=[
            Candidate(material_id="mp-1", formula="AlN", features={"mp_band_gap_ev": 5.7, "nsites": 8}),
            Candidate(material_id="mp-2", formula="GaN", features={"mp_band_gap_ev": 3.2, "nsites": 8}),
            Candidate(material_id="mp-3", formula="Fe2VAl", features={"mp_band_gap_ev": None, "nsites": 12}),
        ],
        goal="recalculate the top 2 candidates with MatGL",
        calculate_matgl=True,
        recalculate_top_n=2,
        matgl_max_recalc_entries=2,
        matgl_max_atoms=50,
    )

    out = predictor.run(payload)

    assert out.provenance.output_summary["forced_matgl_count"] == 2
    assert out.provenance.output_summary["fallback_count"] >= 1
    by_id = {prediction.candidate.material_id: prediction for prediction in out.predictions}
    assert by_id["mp-3"].candidate.features["band_gap_source"] == "fallback"
    assert by_id["mp-3"].predicted.backend.startswith("m3gnet_structure_fallback:")


def test_predict_with_matgl_model_uses_compat_when_primary_path_fails(monkeypatch):
    class BrokenModel:
        def predict_structure(self, _structure):
            raise RuntimeError("primary path failed")

    monkeypatch.setattr(
        predictor_module,
        "_load_matgl_bandgap_model",
        lambda: (BrokenModel(), "MEGNet-MP-2019.4.1-BandGap-mfi", None),
    )
    monkeypatch.setattr(
        predictor_module,
        "_predict_with_matgl_compat",
        lambda _model, _structure: (2.35, None),
    )

    band_gap_ev, err = predictor_module._predict_with_matgl_model(structure={})
    assert err is None
    assert band_gap_ev == 2.35
