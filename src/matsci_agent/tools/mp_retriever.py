from __future__ import annotations

import os
import re
from collections import Counter
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
    use_live_if_available: bool = True
    request_limit_multiplier: int = 4


class MPRetriever:
    """Materials Project retriever interface."""

    def __init__(self, config: MPRetrieverConfig | None = None) -> None:
        self.config = config or MPRetrieverConfig()

    def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
        live: MPRetrieverOutput | None = None
        if self.config.use_live_if_available:
            live = self._retrieve_from_mp(payload)
            if live is not None and live.candidates:
                return live

        source = "mock_fallback_no_live_results" if live is not None else "mock_fallback"
        return self._retrieve_from_mock(payload, source=source)

    def _retrieve_from_mp(self, payload: MPRetrieverInput) -> MPRetrieverOutput | None:
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            return None

        try:
            from mp_api.client import MPRester
        except Exception:
            return None

        limit = payload.limit_override or (payload.constraints.top_k * self.config.request_limit_multiplier)
        excluded_ids = set(payload.exclude_material_ids)
        search_kwargs = self._build_search_kwargs(payload, limit=limit)
        effective = self._effective_filters(payload.constraints, payload.research_goal)
        try:
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(**search_kwargs)
        except Exception:
            return None

        candidates: list[Candidate] = []
        for doc in docs:
            material_id = str(doc.material_id)
            if material_id in excluded_ids:
                continue
            formula = str(doc.formula_pretty)
            doc_elements = {str(element).lower() for element in getattr(doc, "elements", [])}
            elements = doc_elements or self._extract_elements(formula)
            mp_band_gap_ev = getattr(doc, "band_gap", None)
            mp_energy_above_hull = getattr(doc, "energy_above_hull", None)
            nsites = getattr(doc, "nsites", None)
            if not self._passes_client_side_filters(
                effective=effective,
                elements=elements,
                mp_band_gap_ev=mp_band_gap_ev,
                mp_energy_above_hull=mp_energy_above_hull,
                nsites=nsites,
                is_metal=getattr(doc, "is_metal", None),
                theoretical=getattr(doc, "theoretical", None),
                deprecated=getattr(doc, "deprecated", None),
            ):
                continue
            if not self._passes_goal_semantics(
                goal=payload.research_goal,
                element_set=elements,
                mp_band_gap_ev=mp_band_gap_ev,
                min_band_gap_ev=payload.constraints.min_band_gap_ev,
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
                        "nsites": nsites,
                        "structure": (
                            doc.structure.as_dict()
                            if getattr(doc, "structure", None) is not None
                            else None
                        ),
                        "is_metal": getattr(doc, "is_metal", None),
                        "theoretical": getattr(doc, "theoretical", None),
                        "deprecated": getattr(doc, "deprecated", None),
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

        candidates.sort(key=lambda candidate: self._candidate_rank_key(candidate, payload.research_goal))
        deduped = self._dedupe_by_formula(candidates, limit=limit)
        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(mode="json"),
            output_summary={
                "candidate_count": len(deduped),
                "source": "materials_project",
                "fallback_used": False,
                "search_kwargs": self._json_safe_search_kwargs(search_kwargs),
            },
        )
        return MPRetrieverOutput(candidates=deduped, provenance=provenance)

    def _retrieve_from_mock(self, payload: MPRetrieverInput, source: str) -> MPRetrieverOutput:
        mock_pool = [
            Candidate(
                material_id="mp-mock-001",
                formula="Al3Mg2",
                source="mock",
                features={"mp_band_gap_ev": 0.1, "mp_energy_above_hull": 0.12, "nsites": 20, "is_metal": True},
            ),
            Candidate(
                material_id="mp-mock-002",
                formula="CuZn",
                source="mock",
                features={"mp_band_gap_ev": 0.0, "mp_energy_above_hull": 0.06, "nsites": 2, "is_metal": True},
            ),
            Candidate(
                material_id="mp-mock-003",
                formula="AlN",
                source="mock",
                features={"mp_band_gap_ev": 5.9, "mp_energy_above_hull": 0.01, "nsites": 8, "is_metal": False},
            ),
            Candidate(
                material_id="mp-mock-010",
                formula="AlN",
                source="mock",
                features={"mp_band_gap_ev": 5.1, "mp_energy_above_hull": 0.02, "nsites": 8, "is_metal": False},
            ),
            Candidate(
                material_id="mp-mock-004",
                formula="SiC",
                source="mock",
                features={"mp_band_gap_ev": 2.4, "mp_energy_above_hull": 0.02, "nsites": 8, "is_metal": False},
            ),
            Candidate(
                material_id="mp-mock-005",
                formula="Fe2VAl",
                source="mock",
                features={"mp_band_gap_ev": None, "mp_energy_above_hull": None, "nsites": 12, "is_metal": None},
            ),
            Candidate(
                material_id="mp-mock-006",
                formula="Ni3Al",
                source="mock",
                features={"mp_band_gap_ev": None, "mp_energy_above_hull": None, "nsites": 54, "is_metal": None},
            ),
            Candidate(
                material_id="mp-mock-007",
                formula="CoTi",
                source="mock",
                features={"mp_band_gap_ev": 1.0, "mp_energy_above_hull": 0.03, "nsites": 4, "is_metal": False},
            ),
            Candidate(
                material_id="mp-mock-008",
                formula="O2",
                source="mock",
                features={"mp_band_gap_ev": 1.3, "mp_energy_above_hull": 0.00, "nsites": 2, "is_metal": False},
            ),
            Candidate(
                material_id="mp-mock-009",
                formula="AcF3",
                source="mock",
                features={"mp_band_gap_ev": 6.2, "mp_energy_above_hull": 0.02, "nsites": 16, "is_metal": False},
            ),
        ]

        excluded_ids = set(payload.exclude_material_ids)
        limit = payload.limit_override or (payload.constraints.top_k * self.config.request_limit_multiplier)
        effective = self._effective_filters(payload.constraints, payload.research_goal)
        filtered = []
        for candidate in mock_pool:
            if candidate.material_id in excluded_ids:
                continue
            elements = self._extract_elements(candidate.formula)
            if effective.elements and not set(token.lower() for token in effective.elements).issubset(elements):
                continue
            if set(token.lower() for token in effective.exclude_elements) & elements:
                continue
            if effective.band_gap is not None and candidate.features.get("mp_band_gap_ev") is not None:
                band_gap = float(candidate.features["mp_band_gap_ev"])
                if effective.band_gap.min is not None and band_gap < effective.band_gap.min:
                    continue
                if effective.band_gap.max is not None and band_gap > effective.band_gap.max:
                    continue
            if not self._passes_goal_semantics(
                goal=payload.research_goal,
                element_set=elements,
                mp_band_gap_ev=candidate.features.get("mp_band_gap_ev"),
                min_band_gap_ev=payload.constraints.min_band_gap_ev,
            ):
                continue
            filtered.append(candidate.model_copy(deep=True))

        filtered.sort(key=lambda candidate: self._candidate_rank_key(candidate, payload.research_goal))
        deduped = self._dedupe_by_formula(filtered, limit=limit)
        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(mode="json"),
            output_summary={
                "candidate_count": len(deduped),
                "source": source,
                "fallback_used": True,
                "fallback_reason": (
                    "live_mp_returned_zero_candidates_after_server_and_client_filters"
                    if source == "mock_fallback_no_live_results"
                    else f"missing {self.config.api_key_env_var}, missing mp-api, or MP request error"
                ),
            },
        )
        return MPRetrieverOutput(candidates=deduped, provenance=provenance)

    def _build_search_kwargs(self, payload: MPRetrieverInput, limit: int) -> dict[str, Any]:
        effective = self._effective_filters(payload.constraints, payload.research_goal)
        search_kwargs: dict[str, Any] = {
            "fields": [
                "material_id",
                "formula_pretty",
                "elements",
                "band_gap",
                "energy_above_hull",
                "nsites",
                "structure",
                "is_metal",
                "deprecated",
                "theoretical",
                "symmetry",
            ],
            "num_chunks": 1,
            "chunk_size": max(120, limit * 12),
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

    def _effective_filters(self, constraints, goal: str) -> MPFilters:
        filters = constraints.mp_filters.model_copy(deep=True)
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
        theoretical: bool | None,
        deprecated: bool | None,
    ) -> bool:
        required_elements = {token.lower() for token in effective.elements}
        excluded_elements = {token.lower() for token in effective.exclude_elements}
        if required_elements and not required_elements.issubset(elements):
            return False
        if excluded_elements & elements:
            return False
        if effective.is_metal is not None and is_metal is not None and is_metal != effective.is_metal:
            return False
        if effective.theoretical is not None and theoretical is not None and theoretical != effective.theoretical:
            return False
        if effective.deprecated is not None and deprecated is not None and deprecated != effective.deprecated:
            return False
        if effective.band_gap is not None and mp_band_gap_ev is not None:
            if effective.band_gap.min is not None and float(mp_band_gap_ev) < effective.band_gap.min:
                return False
            if effective.band_gap.max is not None and float(mp_band_gap_ev) > effective.band_gap.max:
                return False
        if effective.energy_above_hull is not None and mp_energy_above_hull is not None:
            if effective.energy_above_hull.min is not None and float(mp_energy_above_hull) < effective.energy_above_hull.min:
                return False
            if effective.energy_above_hull.max is not None and float(mp_energy_above_hull) > effective.energy_above_hull.max:
                return False
        if effective.num_sites is not None and nsites is not None:
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

    def _dedupe_by_formula(self, candidates: list[Candidate], limit: int) -> list[Candidate]:
        counts = Counter(candidate.formula for candidate in candidates)
        seen_formulas: set[str] = set()
        deduped: list[Candidate] = []
        for candidate in candidates:
            if candidate.formula in seen_formulas:
                continue
            seen_formulas.add(candidate.formula)
            entry_count = counts[candidate.formula]
            candidate.entry_count = entry_count
            candidate.has_multiple_entries = entry_count > 1
            candidate.features["has_multiple_entries"] = candidate.has_multiple_entries
            candidate.features["entry_count"] = candidate.entry_count
            deduped.append(candidate)
            if len(deduped) >= limit:
                break
        return deduped

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

    @staticmethod
    def _passes_goal_semantics(
        goal: str,
        element_set: set[str],
        mp_band_gap_ev: float | None,
        min_band_gap_ev: float | None,
    ) -> bool:
        if min_band_gap_ev is not None and mp_band_gap_ev is not None and float(mp_band_gap_ev) < min_band_gap_ev:
            return False
        return True
