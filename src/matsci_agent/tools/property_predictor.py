from __future__ import annotations

import math
import os
from typing import Any

from matsci_agent.schemas import (
    PredictedProperties,
    PropertyPredictionRecord,
    PropertyPredictorInput,
    PropertyPredictorOutput,
    ToolCallProvenance,
)

_DEFAULT_MODEL = "models/pretrained/MEGNet-BandGap-mfi-MP-2019.4.1"
_MODEL: Any | None = None
_MODEL_ERROR: str | None = None


class MatGLCalculationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _load_model() -> tuple[Any | None, str | None]:
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL, _MODEL_ERROR
    try:
        import matgl

        model_name = os.getenv("MATSCI_MATGL_MODEL", _DEFAULT_MODEL)
        _MODEL = matgl.load_model(model_name)
    except Exception as exc:
        _MODEL_ERROR = f"MatGL model unavailable: {exc}"
    return _MODEL, _MODEL_ERROR


def _structure(features: dict[str, Any]) -> Any | None:
    value = features.get("structure")
    if value is None:
        return None
    if hasattr(value, "lattice") and hasattr(value, "sites"):
        return value
    if isinstance(value, dict):
        try:
            from pymatgen.core import Structure

            return Structure.from_dict(value)
        except Exception:
            return None
    return None


def _gap(value: Any) -> float | None:
    if isinstance(value, dict):
        value = next((value[key] for key in ("band_gap", "band_gap_ev", "gap") if value.get(key) is not None), None)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None
    try:
        gap = float(value)
    except (TypeError, ValueError):
        return None
    return gap if math.isfinite(gap) and gap >= 0 else None


def _calculate_matgl(features: dict[str, Any], *, relax: bool, max_steps: int) -> tuple[float | None, str | None]:
    structure = _structure(features)
    if structure is None:
        return None, "MP structure is missing or invalid."
    model, error = _load_model()
    if model is None:
        return None, error
    try:
        if relax:
            import matgl
            from matgl.ext.ase import Relaxer

            relax_name = os.getenv("MATSCI_MATGL_RELAX_MODEL", "models/pretrained/TensorNet-PES-MatPES-PBE-2025.2")
            result = Relaxer(potential=matgl.load_model(relax_name)).relax(structure, steps=max_steps)
            structure = result["final_structure"]
        if hasattr(model, "predict_structure"):
            if getattr(getattr(model, "embedding", None), "layer_state_embedding", None) is not None:
                import torch

                # The bundled multifidelity MEGNet uses state 0 for the MP PBE gap.
                output = model.predict_structure(structure, state_attr=torch.tensor([0], dtype=torch.long))
            else:
                output = model.predict_structure(structure)
        elif hasattr(model, "predict"):
            output = model.predict(structure)
        else:
            output = model(structure)
    except Exception as exc:
        return None, f"MatGL calculation failed: {exc}"
    gap = _gap(output)
    return (gap, None) if gap is not None else (None, "MatGL output has no valid band gap.")


class PropertyPredictor:
    def run(self, payload: PropertyPredictorInput) -> PropertyPredictorOutput:
        if payload.calculate_matgl and len(payload.candidates) > payload.matgl_max_recalc_entries:
            raise MatGLCalculationError(
                "matgl_limit_exceeded",
                f"MatGL requested for {len(payload.candidates)} structures; limit is {payload.matgl_max_recalc_entries}.",
            )

        predictions: list[PropertyPredictionRecord] = []
        failures = 0
        for candidate in payload.candidates:
            mp_gap = _gap(candidate.features.get("mp_band_gap_ev"))
            candidate.features["band_gap_source"] = "materials_project" if mp_gap is not None else "unknown"
            candidate.features.pop("matgl_band_gap_ev", None)
            candidate.features.pop("matgl_error", None)
            candidate.features.pop("gap_difference_ev", None)
            predicted = PredictedProperties(band_gap_ev=mp_gap, backend=candidate.features["band_gap_source"])
            if payload.calculate_matgl:
                nsites = candidate.features.get("nsites")
                if not isinstance(nsites, (int, float)) or nsites >= payload.matgl_max_atoms:
                    gap, error = None, f"Structure exceeds the {payload.matgl_max_atoms} atom limit or atom count is missing."
                else:
                    gap, error = _calculate_matgl(
                        candidate.features,
                        relax=payload.enable_relaxation,
                        max_steps=payload.relaxation_max_steps,
                    )
                if error is not None:
                    failures += 1
                    candidate.features["matgl_error"] = error
                    candidate.features["band_gap_source"] = "matgl_failed"
                    predicted = PredictedProperties(backend="matgl_failed", error=error)
                else:
                    candidate.features["matgl_band_gap_ev"] = gap
                    candidate.features["band_gap_source"] = "matgl"
                    if mp_gap is not None:
                        candidate.features["gap_difference_ev"] = round(gap - mp_gap, 6)
                    predicted = PredictedProperties(band_gap_ev=gap, backend="matgl")
            predictions.append(PropertyPredictionRecord(candidate=candidate, predicted=predicted))

        provenance = ToolCallProvenance(
            tool_name="property_predictor",
            input_payload={"candidate_ids": [candidate.material_id for candidate in payload.candidates]},
            output_summary={
                "source": "matgl" if payload.calculate_matgl else "materials_project",
                "prediction_count": len(predictions),
                "matgl_failure_count": failures,
            },
        )
        return PropertyPredictorOutput(predictions=predictions, provenance=provenance)
