from __future__ import annotations

from matsci_agent.schemas import (
    PredictedProperties,
    PropertyPredictionRecord,
    PropertyPredictorInput,
    PropertyPredictorOutput,
    ToolCallProvenance,
)


def composition_predict_crabnet(formula: str, goal: str) -> PredictedProperties:
    """Stage-1 fast surrogate (CrabNet-style placeholder)."""
    base = float(sum(ord(ch) for ch in formula) % 250)
    goal_boost = 30.0 if "thermal" in goal.lower() else 0.0
    conductivity = 20.0 + base + goal_boost
    uncertainty = max(1.0, 20.0 - len(formula))
    return PredictedProperties(
        thermal_conductivity=conductivity,
        uncertainty=uncertainty,
        backend="crabnet_composition_fast",
    )


def composition_predict_roost(formula: str, goal: str) -> PredictedProperties:
    """Stage-1 composition alternative.

    TODO: implement Roost inference.
    """
    del formula, goal
    raise NotImplementedError("Roost composition surrogate is not implemented yet.")


def structure_predict_chgnet(formula: str, goal: str) -> PredictedProperties:
    """Stage-2 accurate surrogate placeholder.

    TODO: implement CHGNet structure-conditioned prediction.
    """
    del formula, goal
    raise NotImplementedError("CHGNet structure surrogate is not implemented yet.")


def structure_predict_m3gnet(formula: str, goal: str) -> PredictedProperties:
    """Stage-2 accurate surrogate placeholder.

    TODO: implement M3GNet structure-conditioned prediction.
    """
    del formula, goal
    raise NotImplementedError("M3GNet structure surrogate is not implemented yet.")


def structure_predict_alignn(formula: str, goal: str) -> PredictedProperties:
    """Stage-2 accurate surrogate placeholder.

    TODO: implement ALIGNN structure-conditioned prediction.
    """
    del formula, goal
    raise NotImplementedError("ALIGNN structure surrogate is not implemented yet.")


class PropertyPredictor:
    def _predict_fast(self, formula: str, goal: str) -> PredictedProperties:
        return composition_predict_crabnet(formula=formula, goal=goal)

    def _predict_accurate(self, formula: str, goal: str) -> PredictedProperties:
        # Default accurate path is CHGNet; other structure models are available placeholders.
        return structure_predict_chgnet(formula=formula, goal=goal)

    def run(self, payload: PropertyPredictorInput) -> PropertyPredictorOutput:
        if payload.surrogate_mode == "fast":
            predict_fn = self._predict_fast
            backend_name = "crabnet_composition_fast"
        else:
            predict_fn = self._predict_accurate
            backend_name = "chgnet_structure_accurate"

        predictions = [
            PropertyPredictionRecord(
                candidate=candidate,
                predicted=predict_fn(candidate.formula, payload.goal),
            )
            for candidate in payload.candidates
        ]

        provenance = ToolCallProvenance(
            tool_name="property_predictor",
            input_payload={
                "goal": payload.goal,
                "surrogate_mode": payload.surrogate_mode,
                "candidate_ids": [c.material_id for c in payload.candidates],
            },
            output_summary={
                "backend": backend_name,
                "surrogate_mode": payload.surrogate_mode,
                "prediction_count": len(predictions),
            },
        )
        return PropertyPredictorOutput(predictions=predictions, provenance=provenance)
