import pytest

from matsci_agent.schemas import Candidate, PropertyPredictorInput
from matsci_agent.tools.property_predictor import PropertyPredictor


def test_fast_mode_uses_crabnet_surrogate():
    predictor = PropertyPredictor()
    payload = PropertyPredictorInput(
        candidates=[Candidate(material_id="c1", formula="Al3Mg2")],
        goal="high thermal conductivity",
        surrogate_mode="fast",
    )

    out = predictor.run(payload)

    assert out.predictions[0].predicted.backend == "crabnet_composition_fast"
    assert out.provenance.output_summary["surrogate_mode"] == "fast"


def test_accurate_mode_is_placeholder_until_structure_model_is_integrated():
    predictor = PropertyPredictor()
    payload = PropertyPredictorInput(
        candidates=[Candidate(material_id="c2", formula="SiC")],
        goal="high thermal conductivity",
        surrogate_mode="accurate",
    )

    with pytest.raises(NotImplementedError, match="CHGNet structure surrogate"):
        predictor.run(payload)
