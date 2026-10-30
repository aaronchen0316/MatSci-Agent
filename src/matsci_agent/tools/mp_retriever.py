from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from matsci_agent.schemas import Candidate, FloatRange, IntRange, MPFilters, MPRetrieverInput, MPRetrieverOutput, ToolCallProvenance

_FLOAT_RANGE_DEFAULTS: dict[str, tuple[float, float]] = {
    "band_gap": (0.0, 20.0),
    "energy_above_hull": (0.0, 5.0),
    "formation_energy": (-20.0, 20.0),
    "density": (0.0, 100.0),
    "efermi": (-30.0, 30.0),
    "total_magnetization": (0.0, 1000.0),
    "volume": (0.0, 10000.0),
}
_INT_RANGE_DEFAULTS: dict[str, tuple[int, int]] = {
    "num_sites": (1, 2000),
    "num_elements": (1, 20),
}


@dataclass
class MPRetrieverConfig:
    api_key_env_var: str = "MP_API_KEY"
    request_limit_multiplier: int = 4
    max_documents: int = 120
    max_searches: int = 3


class MPRetrieverError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MPRetriever:
    """Materials Project retriever interface."""

    def __init__(self, config: MPRetrieverConfig | None = None) -> None:
        self.config = config or MPRetrieverConfig()

    @staticmethod
    def passes_constraints(candidate: Candidate, constraints) -> bool:
        effective = MPRetriever._effective_filters(constraints, "")
        features = candidate.features
        elements = features.get("elements")
        element_set = {str(value).lower() for value in elements} if isinstance(elements, list) and elements else MPRetriever._extract_elements(candidate.formula)
        return MPRetriever._passes_client_side_filters(
            effective=effective,
            elements=element_set,
            mp_band_gap_ev=features.get("mp_band_gap_ev"),
            mp_energy_above_hull=features.get("mp_energy_above_hull"),
            nsites=features.get("nsites"),
            is_metal=features.get("is_metal"),
            is_stable=features.get("is_stable"),
            is_gap_direct=features.get("is_gap_direct"),
            theoretical=features.get("theoretical"),
            deprecated=features.get("deprecated"),
            has_reconstructed=features.get("has_reconstructed"),
            generic_properties=features,
        )

    def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
        return self._retrieve_from_mp(payload)

    def _retrieve_from_mp(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            raise MPRetrieverError("mp_credentials_missing", f"Missing {self.config.api_key_env_var}.")

        try:
            from mp_api.client import MPRester
        except Exception as exc:
            raise MPRetrieverError("mp_dependency_missing", "Materials Project client is unavailable.") from exc

        limit = payload.limit_override or (payload.constraints.top_k * self.config.request_limit_multiplier)
        excluded_ids = set(payload.exclude_material_ids)
        effective = self._effective_filters(payload.constraints, payload.research_goal)
        search_summaries: list[dict[str, Any]] = []
        raw_count = 0
        seen_docs: set[str] = set()

        def _search(search_kwargs: dict[str, Any]) -> list[Candidate]:
            nonlocal raw_count
            remaining = self.config.max_documents - len(seen_docs)
            if remaining <= 0 or len(search_summaries) >= self.config.max_searches:
                return []
            search_kwargs["num_chunks"] = 1
            search_kwargs["chunk_size"] = min(search_kwargs["chunk_size"], remaining)
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(**search_kwargs)
            search_summaries.append(self._json_safe_search_kwargs(search_kwargs))
            candidates: list[Candidate] = []
            for doc in docs[:search_kwargs["chunk_size"]]:
                material_id = str(doc.material_id)
                if material_id in seen_docs:
                    continue
                seen_docs.add(material_id)
                raw_count += 1
                if material_id in excluded_ids:
                    continue
                formula = str(doc.formula_pretty)
                doc_elements = {str(element).lower() for element in getattr(doc, "elements", [])}
                elements = doc_elements or self._extract_elements(formula)
                mp_band_gap_ev = getattr(doc, "band_gap", None)
                mp_energy_above_hull = getattr(doc, "energy_above_hull", None)
                generic_properties = {
                    "formation_energy": getattr(doc, "formation_energy_per_atom", None),
                    "density": getattr(doc, "density", None),
                    "volume": getattr(doc, "volume", None),
                    "efermi": getattr(doc, "efermi", None),
                    "total_magnetization": getattr(doc, "total_magnetization", None),
                }
                nsites = getattr(doc, "nsites", None)
                if not self._passes_client_side_filters(
                    effective=effective,
                    elements=elements,
                    mp_band_gap_ev=mp_band_gap_ev,
                    mp_energy_above_hull=mp_energy_above_hull,
                    nsites=nsites,
                    is_metal=getattr(doc, "is_metal", None),
                    is_stable=getattr(doc, "is_stable", None),
                    is_gap_direct=getattr(doc, "is_gap_direct", None),
                    theoretical=getattr(doc, "theoretical", None),
                    deprecated=getattr(doc, "deprecated", None),
                    has_reconstructed=getattr(doc, "has_reconstructed", None),
                    generic_properties=generic_properties,
                ):
                    continue
                symmetry = getattr(doc, "symmetry", None)
                candidates.append(
                    Candidate(
                        material_id=material_id,
                        formula=formula,
                        source="materials_project",
                        features={
                            "elements": [str(element) for element in getattr(doc, "elements", [])],
                            "mp_band_gap_ev": mp_band_gap_ev,
                            "mp_energy_above_hull": mp_energy_above_hull,
                            **generic_properties,
                            "nsites": nsites,
                            "structure": (
                                doc.structure.as_dict()
                                if getattr(doc, "structure", None) is not None
                                else None
                            ),
                            "is_metal": getattr(doc, "is_metal", None),
                            "is_stable": getattr(doc, "is_stable", None),
                            "is_gap_direct": getattr(doc, "is_gap_direct", None),
                            "theoretical": getattr(doc, "theoretical", None),
                            "deprecated": getattr(doc, "deprecated", None),
                            "has_reconstructed": getattr(doc, "has_reconstructed", None),
                            "crystal_system": (
                                str(getattr(symmetry, "crystal_system", "")).lower()
                                if symmetry is not None and getattr(symmetry, "crystal_system", None) is not None
                                else None
                            ),
                            "spacegroup_symbol": (
                                getattr(symmetry, "symbol", None)
                                if symmetry is not None
                                else None
                            ),
                            "spacegroup_number": (
                                getattr(symmetry, "number", None)
                                if symmetry is not None
                                else None
                            ),
                        },
                    )
                )
            return candidates

        try:
            if effective.material_ids is not None:
                id_kwargs = self._build_search_kwargs(payload, limit=limit)
                candidates = _search(id_kwargs)
            elif payload.search_space_targets:
                formula_kwargs = self._build_search_kwargs(payload, limit=limit)
                formula_kwargs["formula"] = effective.formula or [target.normalized_formula for target in payload.search_space_targets]
                candidates = _search(formula_kwargs)
                if effective.formula is None and len({candidate.formula for candidate in candidates}) < payload.constraints.top_k:
                    chemsys_values = effective.chemsys or sorted({target.chemsys for target in payload.search_space_targets if target.chemsys})
                    if chemsys_values:
                        chemsys_kwargs = self._build_search_kwargs(payload, limit=limit)
                        chemsys_kwargs.pop("formula", None)
                        chemsys_kwargs["chemsys"] = chemsys_values
                        seen = {candidate.material_id for candidate in candidates}
                        for candidate in _search(chemsys_kwargs):
                            if candidate.material_id not in seen:
                                candidates.append(candidate)
                                seen.add(candidate.material_id)
                        extra_formulas = sorted({candidate.formula for candidate in candidates})
                        if extra_formulas:
                            complete_kwargs = self._build_search_kwargs(payload, limit=limit)
                            complete_kwargs.pop("chemsys", None)
                            complete_kwargs["formula"] = extra_formulas
                            for candidate in _search(complete_kwargs):
                                if candidate.material_id not in seen:
                                    candidates.append(candidate)
                                    seen.add(candidate.material_id)
            else:
                search_kwargs = self._build_search_kwargs(payload, limit=limit)
                candidates = _search(search_kwargs)
        except Exception as exc:
            raise MPRetrieverError("mp_service_failed", "Materials Project query failed.") from exc

        candidates.sort(key=lambda candidate: self._candidate_rank_key(candidate, payload.research_goal))
        reason_code = None if candidates else "constraint_no_pass" if raw_count else "no_matches"
        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(mode="json"),
            output_summary={
                "candidate_count": len(candidates),
                "raw_document_count": raw_count,
                "source": "materials_project",
                "fallback_used": False,
                "search_kwargs": search_summaries[0] if search_summaries else {},
                "search_kwargs_sequence": search_summaries,
                "search_space_target_count": len(payload.search_space_targets),
            },
        )
        return MPRetrieverOutput(candidates=candidates, provenance=provenance, reason_code=reason_code)

    def _build_search_kwargs(self, payload: MPRetrieverInput, limit: int) -> dict[str, Any]:
        effective = self._effective_filters(payload.constraints, payload.research_goal)
        search_kwargs: dict[str, Any] = {
            "fields": [
                "material_id",
                "formula_pretty",
                "elements",
                "band_gap",
                "energy_above_hull",
                "formation_energy_per_atom",
                "density",
                "volume",
                "efermi",
                "total_magnetization",
                "nsites",
                "structure",
                "is_metal",
                "is_stable",
                "is_gap_direct",
                "deprecated",
                "theoretical",
                "symmetry",
                "has_reconstructed",
            ],
            "num_chunks": 1,
            "chunk_size": min(self.config.max_documents, max(20, limit)),
        }

        if effective.formula is not None:
            search_kwargs["formula"] = effective.formula
        if effective.chemsys is not None:
            search_kwargs["chemsys"] = effective.chemsys
        if effective.material_ids is not None:
            search_kwargs["material_ids"] = effective.material_ids
        if effective.elements:
            search_kwargs["elements"] = effective.elements
        if effective.exclude_elements:
            search_kwargs["exclude_elements"] = effective.exclude_elements
        if effective.possible_species:
            search_kwargs["possible_species"] = effective.possible_species
        if effective.has_props:
            search_kwargs["has_props"] = effective.has_props
        for field_name in ["is_metal", "is_stable", "is_gap_direct", "theoretical", "deprecated", "has_reconstructed"]:
            value = getattr(effective, field_name)
            if value is not None:
                search_kwargs[field_name] = value
        if effective.crystal_system is not None:
            search_kwargs["crystal_system"] = self._coerce_crystal_system(effective.crystal_system)
        if effective.spacegroup_number is not None:
            search_kwargs["spacegroup_number"] = effective.spacegroup_number
        if effective.spacegroup_symbol is not None:
            search_kwargs["spacegroup_symbol"] = effective.spacegroup_symbol
        for field_name in ["band_gap", "energy_above_hull", "formation_energy", "density", "efermi", "total_magnetization", "volume"]:
            value = getattr(effective, field_name)
            if value is not None:
                search_kwargs[field_name] = self._float_range_to_tuple(field_name, value)
        for field_name in ["num_sites", "num_elements"]:
            value = getattr(effective, field_name)
            if value is not None:
                search_kwargs[field_name] = self._int_range_to_tuple(field_name, value)
        return search_kwargs

    @staticmethod
    def _effective_filters(constraints, goal: str) -> MPFilters:
        filters = constraints.mp_filters.model_copy(deep=True)
        # Band gap is a summary field, not an MP has_props value.
        if "band_gap" in filters.has_props:
            filters.has_props = [prop for prop in filters.has_props if prop != "band_gap"]
            filters.band_gap = filters.band_gap or FloatRange(min=0.0)
        filters.elements = sorted(set(filters.elements) | set(constraints.required_elements))
        filters.exclude_elements = sorted(set(filters.exclude_elements) | set(constraints.banned_elements))
        if constraints.min_band_gap_ev is not None:
            filters.band_gap = filters.band_gap or FloatRange()
            current_min = filters.band_gap.min
            filters.band_gap.min = max(current_min, constraints.min_band_gap_ev) if current_min is not None else constraints.min_band_gap_ev
        if constraints.max_energy_above_hull is not None:
            filters.energy_above_hull = filters.energy_above_hull or FloatRange()
            current_max = filters.energy_above_hull.max
            filters.energy_above_hull.max = min(current_max, constraints.max_energy_above_hull) if current_max is not None else constraints.max_energy_above_hull

        return filters

    @staticmethod
    def _passes_client_side_filters(
        *,
        effective: MPFilters,
        elements: set[str],
        mp_band_gap_ev: float | None,
        mp_energy_above_hull: float | None,
        nsites: int | None,
        is_metal: bool | None,
        is_stable: bool | None = None,
        is_gap_direct: bool | None = None,
        theoretical: bool | None,
        deprecated: bool | None,
        has_reconstructed: bool | None = None,
        generic_properties: dict[str, Any] | None = None,
    ) -> bool:
        required_elements = {token.lower() for token in effective.elements}
        excluded_elements = {token.lower() for token in effective.exclude_elements}
        if required_elements and not required_elements.issubset(elements):
            return False
        if excluded_elements & elements:
            return False
        if effective.is_metal is not None and is_metal != effective.is_metal:
            return False
        if effective.is_stable is not None and is_stable != effective.is_stable:
            return False
        if effective.is_gap_direct is not None and is_gap_direct != effective.is_gap_direct:
            return False
        if effective.theoretical is not None and theoretical != effective.theoretical:
            return False
        if effective.deprecated is not None and deprecated != effective.deprecated:
            return False
        if effective.has_reconstructed is not None and has_reconstructed != effective.has_reconstructed:
            return False
        if effective.band_gap is not None:
            if mp_band_gap_ev is None:
                return False
            if effective.band_gap.min is not None and float(mp_band_gap_ev) < effective.band_gap.min:
                return False
            if effective.band_gap.max is not None and float(mp_band_gap_ev) > effective.band_gap.max:
                return False
        if effective.energy_above_hull is not None:
            if mp_energy_above_hull is None:
                return False
            if effective.energy_above_hull.min is not None and float(mp_energy_above_hull) < effective.energy_above_hull.min:
                return False
            if effective.energy_above_hull.max is not None and float(mp_energy_above_hull) > effective.energy_above_hull.max:
                return False
        for field_name in ["formation_energy", "density", "efermi", "total_magnetization", "volume"]:
            range_filter = getattr(effective, field_name)
            value = (generic_properties or {}).get(field_name)
            if range_filter is None:
                continue
            if value is None:
                return False
            if range_filter.min is not None and float(value) < range_filter.min:
                return False
            if range_filter.max is not None and float(value) > range_filter.max:
                return False
        if effective.num_sites is not None:
            if nsites is None:
                return False
            if effective.num_sites.min is not None and int(nsites) < effective.num_sites.min:
                return False
            if effective.num_sites.max is not None and int(nsites) > effective.num_sites.max:
                return False
        if effective.num_elements is not None:
            num_elements = len(elements)
            if effective.num_elements.min is not None and num_elements < effective.num_elements.min:
                return False
            if effective.num_elements.max is not None and num_elements > effective.num_elements.max:
                return False
        return True

    @staticmethod
    def _float_range_to_tuple(field_name: str, value: FloatRange) -> tuple[float, float]:
        default_min, default_max = _FLOAT_RANGE_DEFAULTS[field_name]
        lower = value.min if value.min is not None else default_min
        upper = value.max if value.max is not None else default_max
        return (float(lower), float(upper))

    @staticmethod
    def _int_range_to_tuple(field_name: str, value: IntRange) -> tuple[int, int]:
        default_min, default_max = _INT_RANGE_DEFAULTS[field_name]
        lower = value.min if value.min is not None else default_min
        upper = value.max if value.max is not None else default_max
        return (int(lower), int(upper))

    @staticmethod
    def _coerce_crystal_system(value: str):
        try:
            from emmet.core.symmetry import CrystalSystem
        except Exception:
            return value
        normalized = value.strip().lower()
        for member in CrystalSystem:
            if str(member.value).lower() == normalized:
                return member
        return value

    @staticmethod
    def _candidate_rank_key(candidate: Candidate, goal: str) -> tuple[float, str]:
        band_gap = candidate.features.get("mp_band_gap_ev")
        return (-(float(band_gap) if band_gap is not None else -1.0), candidate.material_id)

    @staticmethod
    def _json_safe_search_kwargs(search_kwargs: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in search_kwargs.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            elif isinstance(value, list):
                safe[key] = value
            elif isinstance(value, tuple):
                safe[key] = list(value)
            else:
                safe[key] = str(value)
        return safe

    @staticmethod
    def _extract_elements(formula: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[A-Z][a-z]?", formula)}
