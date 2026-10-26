import pytest
import sys
from types import ModuleType

from matsci_agent.schemas import Candidate, PropertyPredictorInput
from matsci_agent.tools import property_predictor as module
from matsci_agent.tools.property_predictor import MatGLCalculationError, PropertyPredictor


def _candidate(material_id: str, gap: float | None) -> Candidate:
    return Candidate(
        material_id=material_id,
        formula="AlN",
        source="materials_project",
        features={"mp_band_gap_ev": gap, "nsites": 8, "structure": {"test": True}},
    )


def test_without_explicit_matgl_uses_mp_or_unknown(monkeypatch):
    monkeypatch.setattr(module, "_calculate_matgl", lambda *_args, **_kwargs: pytest.fail("MatGL ran implicitly"))
    result = PropertyPredictor().run(PropertyPredictorInput(
        candidates=[_candidate("mp-1", 5.7), _candidate("mp-2", None)],
        goal="find semiconductors",
    ))
    assert [item.predicted.band_gap_ev for item in result.predictions] == [5.7, None]
    assert [item.predicted.backend for item in result.predictions] == ["materials_project", "unknown"]


def test_explicit_matgl_reports_value_and_difference(monkeypatch):
    monkeypatch.setattr(module, "_calculate_matgl", lambda *_args, **_kwargs: (1.5, None))
    result = PropertyPredictor().run(PropertyPredictorInput(
        candidates=[_candidate("mp-1", 2.5)], goal="calculate MatGL", calculate_matgl=True,
    ))
    item = result.predictions[0]
    assert item.predicted.band_gap_ev == 1.5
    assert item.candidate.features["mp_band_gap_ev"] == 2.5
    assert item.candidate.features["gap_difference_ev"] == -1.0


def test_matgl_failure_does_not_substitute_mp_gap(monkeypatch):
    monkeypatch.setattr(module, "_calculate_matgl", lambda *_args, **_kwargs: (None, "model failed"))
    result = PropertyPredictor().run(PropertyPredictorInput(
        candidates=[_candidate("mp-1", 5.7)], goal="calculate MatGL", calculate_matgl=True,
    ))
    item = result.predictions[0]
    assert item.predicted.band_gap_ev is None
    assert item.predicted.error == "model failed"
    assert item.candidate.features["mp_band_gap_ev"] == 5.7
    assert item.candidate.features["matgl_error"] == "model failed"
    assert "matgl_band_gap_ev" not in item.candidate.features


def test_matgl_limit_stops_before_any_calculation(monkeypatch):
    monkeypatch.setattr(module, "_calculate_matgl", lambda *_args, **_kwargs: pytest.fail("Calculation started"))
    with pytest.raises(MatGLCalculationError) as error:
        PropertyPredictor().run(PropertyPredictorInput(
            candidates=[_candidate("mp-1", 2.0), _candidate("mp-2", 3.0)],
            goal="calculate MatGL", calculate_matgl=True, matgl_max_recalc_entries=1,
        ))
    assert error.value.code == "matgl_limit_exceeded"


def test_invalid_matgl_output_is_failure_not_1_2_ev(monkeypatch):
    class Model:
        def predict_structure(self, _structure):
            return {"other_property": 1.2}

    monkeypatch.setattr(module, "_structure", lambda _features: object())
    monkeypatch.setattr(module, "_load_model", lambda: (Model(), None))
    gap, error = module._calculate_matgl({"structure": {}}, relax=False, max_steps=1)
    assert gap is None
    assert "no valid band gap" in error


def test_band_gap_model_load_does_not_switch_graph_backend(monkeypatch):
    matgl = ModuleType("matgl")
    matgl.load_model = lambda _name: "pyg model"
    matgl.set_backend = lambda _name: pytest.fail("Backend switch is unsupported")
    monkeypatch.setitem(sys.modules, "matgl", matgl)
    monkeypatch.setattr(module, "_MODEL", None)
    monkeypatch.setattr(module, "_MODEL_ERROR", None)

    assert module._load_model() == ("pyg model", None)


def test_relaxation_uses_pyg_relaxer_with_pyg_potential(monkeypatch):
    class Model:
        def predict_structure(self, structure):
            assert structure == "relaxed"
            return 2.1

    class Relaxer:
        def __init__(self, potential):
            assert potential == "pyg potential"

        def relax(self, structure, steps):
            assert structure == "original"
            assert steps == 3
            return {"final_structure": "relaxed"}

    matgl = ModuleType("matgl")
    matgl.load_model = lambda _name: "pyg potential"
    matgl_ext = ModuleType("matgl.ext")
    matgl_ext.__path__ = []
    pyg_ase = ModuleType("matgl.ext.ase")
    pyg_ase.Relaxer = Relaxer
    monkeypatch.setitem(sys.modules, "matgl", matgl)
    monkeypatch.setitem(sys.modules, "matgl.ext", matgl_ext)
    monkeypatch.setitem(sys.modules, "matgl.ext.ase", pyg_ase)
    monkeypatch.setattr(module, "_structure", lambda _features: "original")
    monkeypatch.setattr(module, "_load_model", lambda: (Model(), None))

    assert module._calculate_matgl({"structure": {}}, relax=True, max_steps=3) == (2.1, None)
