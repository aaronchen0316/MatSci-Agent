from __future__ import annotations

from abc import ABC, abstractmethod

from matsci_agent.schemas import (
    PredictedProperties,
    PropertyPredictionRecord,
    PropertyPredictorInput,
    PropertyPredictorOutput,
    ToolCallProvenance,
)


class PredictorBackend(ABC):
    name: str

    @abstractmethod
    def predict(self, formula: str, goal: str) -> PredictedProperties:
        raise NotImplementedError


class MockCompositionBackend(PredictorBackend):
    """Stub for stage-1 composition model.

    TODO: plug in CrabNet/Roost inference.
    """

    name = "mock_composition_backend"

    def predict(self, formula: str, goal: str) -> PredictedProperties:
        base = float(sum(ord(ch) for ch in formula) % 250)
        goal_boost = 30.0 if "thermal" in goal.lower() else 0.0
        conductivity = 20.0 + base + goal_boost
        uncertainty = max(1.0, 20.0 - len(formula))
        return PredictedProperties(
            thermal_conductivity=conductivity,
            uncertainty=uncertainty,
            backend=self.name,
        )


class PropertyPredictor:
    def __init__(self, backend: PredictorBackend | None = None) -> None:
        self.backend = backend or MockCompositionBackend()

    def run(self, payload: PropertyPredictorInput) -> PropertyPredictorOutput:
        predictions = [
            PropertyPredictionRecord(
                candidate=candidate,
                predicted=self.backend.predict(candidate.formula, payload.goal),
            )
            for candidate in payload.candidates
        ]

        provenance = ToolCallProvenance(
            tool_name="property_predictor",
            input_payload={
                "goal": payload.goal,
                "candidate_ids": [c.material_id for c in payload.candidates],
            },
            output_summary={
                "backend": self.backend.name,
                "prediction_count": len(predictions),
            },
        )
        return PropertyPredictorOutput(predictions=predictions, provenance=provenance)
