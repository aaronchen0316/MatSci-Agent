from __future__ import annotations

import os
from typing import Any

from matsci_agent.schemas import (
    PredictedProperties,
    PropertyPredictionRecord,
    PropertyPredictorInput,
    PropertyPredictorOutput,
    ToolCallProvenance,
)


_M3GNET_MODEL: Any | None = None
_M3GNET_LOAD_ERROR: str | None = None
_MATGL_MODEL: Any | None = None
_MATGL_MODEL_NAME: str = ""
_MATGL_LOAD_ERROR: str | None = None
_MATGL_RELAXER: Any | None = None
_MATGL_RELAXER_ERROR: str | None = None


def _heuristic_fallback(formula: str, goal: str, reason: str) -> PredictedProperties:
    base = float(sum((i + 1) * ord(ch) for i, ch in enumerate(formula)) % 300)
    goal_boost = 1.5 if "band gap" in goal.lower() or "semiconductor" in goal.lower() else 0.5
    band_gap_ev = 0.5 + (base / 120.0) + goal_boost
    uncertainty = max(6.0, 16.0 - 0.2 * len(formula))
    return PredictedProperties(
        band_gap_ev=band_gap_ev,
        uncertainty=uncertainty,
        backend=f"m3gnet_structure_fallback:{reason}",
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_float(value[0])
    for attr in ("item",):
        if hasattr(value, attr):
            try:
                return float(getattr(value, attr)())
            except Exception:
                pass
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            arr = value.detach().cpu().numpy().reshape(-1)
            if arr.size:
                return float(arr[0])
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return None


def _extract_band_gap_ev(model_output: Any) -> float | None:
    raw = None
    if isinstance(model_output, dict):
        raw = (
            model_output.get("band_gap")
            or model_output.get("band_gap_ev")
            or model_output.get("gap")
        )
    elif hasattr(model_output, "get"):
        try:
            raw = (
                model_output.get("band_gap")
                or model_output.get("band_gap_ev")
                or model_output.get("gap")
            )
        except Exception:
            raw = None
    if raw is None:
        raw = model_output
    parsed = _coerce_float(raw)
    if parsed is None:
        return None
    return max(0.0, parsed)


def _estimate_band_gap_ev_from_model_output(model_output: Any) -> float:
    value = _extract_band_gap_ev(model_output)
    if value is None:
        return 1.2
    return value


def _matgl_model_candidates() -> list[str]:
    candidates: list[str] = []
    direct = os.getenv("MATSCI_MATGL_MODEL", "").strip()
    if direct:
        candidates.append(direct)
    extra = os.getenv("MATSCI_MATGL_MODEL_CANDIDATES", "").strip()
    if extra:
        candidates.extend([token.strip() for token in extra.split(",") if token.strip()])
    # Known/likely names across MatGL model registries.
    candidates.extend(
        [
            "MEGNet-MP-2018.6.1-BandGap-mfi",
            "MEGNet-MP-2019.4.1-BandGap-mfi",
            "band_gap",
        ]
    )
    dedup: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if name not in seen:
            dedup.append(name)
            seen.add(name)
    return dedup


def _load_matgl_bandgap_model() -> tuple[Any | None, str, str | None]:
    global _MATGL_MODEL, _MATGL_MODEL_NAME, _MATGL_LOAD_ERROR
    if _MATGL_MODEL is not None:
        return _MATGL_MODEL, _MATGL_MODEL_NAME, None
    if _MATGL_LOAD_ERROR is not None:
        return None, "", _MATGL_LOAD_ERROR

    try:
        import matgl
    except Exception as exc:
        _MATGL_LOAD_ERROR = f"matgl_import_failed:{exc}"
        return None, "", _MATGL_LOAD_ERROR

    errors: list[str] = []
    for candidate_name in _matgl_model_candidates():
        try:
            _MATGL_MODEL = matgl.load_model(candidate_name)
            _MATGL_MODEL_NAME = candidate_name
            return _MATGL_MODEL, _MATGL_MODEL_NAME, None
        except Exception as exc:
            errors.append(f"{candidate_name}:{exc}")

    _MATGL_LOAD_ERROR = "; ".join(errors) if errors else "matgl_model_not_found"
    return None, "", _MATGL_LOAD_ERROR


def _predict_with_matgl_model(structure: Any) -> tuple[float | None, str | None]:
    model, _model_name, load_error = _load_matgl_bandgap_model()
    if model is None:
        return None, load_error

    try:
        if hasattr(model, "predict_structure"):
            output = model.predict_structure(structure)
        elif hasattr(model, "predict"):
            try:
                output = model.predict(structure)
            except Exception:
                output = model.predict([structure])
        elif callable(model):
            try:
                output = model(structure)
            except Exception:
                output = model([structure])
        else:
            return None, "matgl_model_has_no_predict_interface"
    except Exception as exc:
        return None, f"matgl_predict_failed:{exc}"

    parsed = _extract_band_gap_ev(output)
    if parsed is None:
        return None, "matgl_output_missing_band_gap"
    return parsed, None


def _load_m3gnet_model() -> tuple[Any | None, str | None]:
    global _M3GNET_MODEL, _M3GNET_LOAD_ERROR
    if _M3GNET_MODEL is not None:
        return _M3GNET_MODEL, None
    if _M3GNET_LOAD_ERROR is not None:
        return None, _M3GNET_LOAD_ERROR

    try:
        import torch

        # Torch-hub fallback path kept for environments without matgl package.
        try:
            model_names = torch.hub.list("materialsvirtuallab/matgl")
        except Exception:
            model_names = []
        candidate_names = [
            name
            for name in model_names
            if "band" in name.lower() and "gap" in name.lower()
        ]
        model_name = candidate_names[0] if candidate_names else "m3gnet_band_gap"
        _M3GNET_MODEL = torch.hub.load("materialsvirtuallab/matgl", model_name)
        return _M3GNET_MODEL, None
    except Exception as exc:
        _M3GNET_LOAD_ERROR = str(exc)
        return None, _M3GNET_LOAD_ERROR


def _extract_structure(features: dict[str, Any]) -> Any | None:
    structure = features.get("structure")
    if structure is None:
        return None

    if hasattr(structure, "lattice") and hasattr(structure, "sites"):
        return structure

    if isinstance(structure, dict):
        try:
            from pymatgen.core import Structure

            return Structure.from_dict(structure)
        except Exception:
            return None

    return None


def _load_matgl_relaxer() -> tuple[Any | None, str | None]:
    global _MATGL_RELAXER, _MATGL_RELAXER_ERROR
    if _MATGL_RELAXER is not None:
        return _MATGL_RELAXER, None
    if _MATGL_RELAXER_ERROR is not None:
        return None, _MATGL_RELAXER_ERROR

    relax_model_name = os.getenv("MATSCI_MATGL_RELAX_MODEL", "M3GNet-MP-2021.2.8-PES")
    try:
        import matgl
        from matgl.ext.ase import Relaxer
    except Exception as exc:
        _MATGL_RELAXER_ERROR = f"matgl_relax_import_failed:{exc}"
        return None, _MATGL_RELAXER_ERROR

    try:
        potential = matgl.load_model(relax_model_name)
        try:
            _MATGL_RELAXER = Relaxer(potential=potential)
        except Exception:
            _MATGL_RELAXER = Relaxer(potential)
        return _MATGL_RELAXER, None
    except Exception as exc:
        _MATGL_RELAXER_ERROR = f"matgl_relaxer_init_failed:{exc}"
        return None, _MATGL_RELAXER_ERROR


def _maybe_relax_structure(
    structure: Any, enable_relaxation: bool, max_steps: int
) -> tuple[Any, bool]:
    if not enable_relaxation:
        return structure, False

    try:
        relaxer, err = _load_matgl_relaxer()
        if relaxer is None:
            _ = err
            return structure, False

        if hasattr(relaxer, "relax"):
            try:
                result = relaxer.relax(structure, steps=max_steps)
            except TypeError:
                result = relaxer.relax(structure)
        else:
            return structure, False

        if isinstance(result, dict):
            relaxed = (
                result.get("final_structure")
                or result.get("relaxed_structure")
                or result.get("structure")
            )
        else:
            relaxed = result
        if relaxed is not None and hasattr(relaxed, "lattice") and hasattr(relaxed, "sites"):
            return relaxed, True
        return structure, False
    except Exception:
        return structure, False


def structure_predict_m3gnet(
    formula: str,
    goal: str,
    features: dict[str, Any] | None = None,
    enable_relaxation: bool = False,
    relaxation_max_steps: int = 200,
) -> PredictedProperties:
    """Band-gap prediction path with MatGL-first and torch-hub fallback."""
    features = features or {}
    structure = _extract_structure(features)
    if structure is None:
        return _heuristic_fallback(formula, goal, reason="missing_structure")

    structure, was_relaxed = _maybe_relax_structure(
        structure,
        enable_relaxation=enable_relaxation,
        max_steps=relaxation_max_steps,
    )

    matgl_gap_ev, matgl_err = _predict_with_matgl_model(structure)
    if matgl_gap_ev is not None:
        return PredictedProperties(
            band_gap_ev=matgl_gap_ev,
            uncertainty=1.0 if was_relaxed else 1.2,
            backend="matgl_band_gap_relaxed" if was_relaxed else "matgl_band_gap",
        )

    model, load_error = _load_m3gnet_model()
    if model is None:
        reason = (
            "matgl_and_m3gnet_unavailable"
            if matgl_err is not None
            else "m3gnet_model_unavailable"
        )
        return _heuristic_fallback(formula, goal, reason=reason)

    try:
        if hasattr(model, "predict_structure"):
            model_output = model.predict_structure(structure)
        else:
            # Some wrappers expose callable model interfaces.
            model_output = model(structure)
    except Exception:
        reason = (
            "matgl_and_m3gnet_inference_error"
            if matgl_err is not None
            else "m3gnet_inference_error"
        )
        return _heuristic_fallback(formula, goal, reason=reason)

    band_gap_ev = _estimate_band_gap_ev_from_model_output(model_output)
    uncertainty = 4.0 if load_error is None else 6.0
    return PredictedProperties(
        band_gap_ev=band_gap_ev,
        uncertainty=uncertainty,
        backend="m3gnet_structure_relaxed" if was_relaxed else "m3gnet_structure",
    )


def _candidate_sort_key(material_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in material_id if ch.isdigit())
    return (int(digits) if digits else 10**12, material_id)


def _is_matgl_atom_eligible(nsites: Any, max_atoms: int) -> bool:
    return isinstance(nsites, (int, float)) and int(nsites) < max_atoms


class PropertyPredictor:
    @staticmethod
    def _select_matgl_indices(payload: PropertyPredictorInput) -> tuple[set[int], set[int]]:
        matgl_needed: list[int] = []
        for idx, candidate in enumerate(payload.candidates):
            mp_gap = candidate.features.get("mp_band_gap_ev")
            if payload.calculate_matgl or mp_gap is None:
                matgl_needed.append(idx)

        eligible = [
            idx
            for idx in matgl_needed
            if _is_matgl_atom_eligible(
                payload.candidates[idx].features.get("nsites"), payload.matgl_max_atoms
            )
        ]
        eligible_sorted = sorted(
            eligible,
            key=lambda i: _candidate_sort_key(payload.candidates[i].material_id),
        )
        selected = set(eligible_sorted[: payload.matgl_max_recalc_entries])
        return selected, set(matgl_needed)

    def run(self, payload: PropertyPredictorInput) -> PropertyPredictorOutput:
        predictions: list[PropertyPredictionRecord] = []
        fallback_count = 0
        used_mp_count = 0
        used_matgl_count = 0
        forced_matgl_count = 0
        matgl_skipped_count = 0
        matgl_selected_indices, matgl_needed_indices = self._select_matgl_indices(payload)
        selected_material_ids: list[str] = []
        skipped_material_ids: list[str] = []

        for idx, candidate in enumerate(payload.candidates):
            mp_gap = candidate.features.get("mp_band_gap_ev")
            nsites = candidate.features.get("nsites")
            has_mp_gap = mp_gap is not None
            needs_matgl = idx in matgl_needed_indices
            selected_for_matgl = idx in matgl_selected_indices

            candidate.features.pop("matgl_skipped_reason", None)
            candidate.features["matgl_forced"] = bool(payload.calculate_matgl)

            if not needs_matgl:
                predicted = PredictedProperties(
                    band_gap_ev=float(mp_gap),
                    uncertainty=0.2,
                    backend="materials_project_band_gap",
                )
                used_mp_count += 1
                candidate.features["band_gap_source"] = "materials_project"
                predictions.append(
                    PropertyPredictionRecord(candidate=candidate, predicted=predicted)
                )
                continue

            if selected_for_matgl:
                predicted = structure_predict_m3gnet(
                    formula=candidate.formula,
                    goal=payload.goal,
                    features=candidate.features,
                    enable_relaxation=payload.enable_relaxation,
                    relaxation_max_steps=payload.relaxation_max_steps,
                )
                selected_material_ids.append(candidate.material_id)
                if payload.calculate_matgl:
                    forced_matgl_count += 1
                if predicted.backend.startswith("m3gnet_structure_fallback:"):
                    fallback_count += 1
                    candidate.features["band_gap_source"] = "fallback"
                else:
                    used_matgl_count += 1
                    candidate.features["band_gap_source"] = "matgl"
            else:
                matgl_skipped_count += 1
                skipped_material_ids.append(candidate.material_id)
                is_atom_count_eligible = _is_matgl_atom_eligible(
                    nsites, payload.matgl_max_atoms
                )
                reason = (
                    "atoms_too_high_or_missing_nsites"
                    if not is_atom_count_eligible
                    else "recalc_limit_reached"
                )
                candidate.features["matgl_skipped_reason"] = reason
                if has_mp_gap:
                    predicted = PredictedProperties(
                        band_gap_ev=float(mp_gap),
                        uncertainty=0.4,
                        backend="materials_project_band_gap",
                    )
                    used_mp_count += 1
                    candidate.features["band_gap_source"] = "materials_project"
                else:
                    predicted = _heuristic_fallback(candidate.formula, payload.goal, reason)
                    fallback_count += 1
                    candidate.features["band_gap_source"] = "fallback"

            predictions.append(
                PropertyPredictionRecord(candidate=candidate, predicted=predicted)
            )

        provenance = ToolCallProvenance(
            tool_name="property_predictor",
            input_payload={
                "goal": payload.goal,
                "candidate_ids": [c.material_id for c in payload.candidates],
            },
            output_summary={
                "backend": "hybrid_mp_then_matgl",
                "prediction_count": len(predictions),
                "fallback_count": fallback_count,
                "used_mp_count": used_mp_count,
                "used_matgl_count": used_matgl_count,
                "forced_matgl_count": forced_matgl_count,
                "matgl_skipped_count": matgl_skipped_count,
                "matgl_selected_material_ids": selected_material_ids,
                "matgl_skipped_material_ids": skipped_material_ids,
                "relaxation_enabled": payload.enable_relaxation,
            },
        )
        return PropertyPredictorOutput(predictions=predictions, provenance=provenance)
