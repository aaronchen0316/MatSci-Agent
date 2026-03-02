from __future__ import annotations

import os
import re
from dataclasses import dataclass

from matsci_agent.schemas import (
    Candidate,
    MPRetrieverInput,
    MPRetrieverOutput,
    ToolCallProvenance,
)


@dataclass
class MPRetrieverConfig:
    api_key_env_var: str = "MP_API_KEY"
    use_live_if_available: bool = True
    request_limit_multiplier: int = 2


class MPRetriever:
    """Materials Project retriever interface.

    Uses live MP retrieval when API key + client are available, and
    gracefully falls back to mock candidates otherwise.
    """

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
            # Optional import so local development works without mp-api installed.
            from mp_api.client import MPRester
        except Exception:
            return None

        banned = {el.lower() for el in payload.constraints.banned_elements}
        required = {el.lower() for el in payload.constraints.required_elements}
        limit = payload.constraints.top_k * self.config.request_limit_multiplier
        goal_lower = payload.research_goal.lower()
        is_semiconductor_goal = "semiconductor" in goal_lower
        band_gap_min = payload.constraints.min_band_gap_ev
        if band_gap_min is None and is_semiconductor_goal:
            # Avoid metallic/near-zero-gap first-page hits when user asks for semiconductors.
            band_gap_min = 0.01

        search_kwargs: dict[str, object] = {
            "elements": payload.constraints.required_elements or None,
            "exclude_elements": payload.constraints.banned_elements or None,
            "fields": [
                "material_id",
                "formula_pretty",
                "elements",
                "band_gap",
                "nsites",
                "structure",
            ],
            "num_chunks": 1,
            "chunk_size": max(120, limit * 12),
        }
        if band_gap_min is not None:
            search_kwargs["band_gap"] = (float(band_gap_min), 20.0)
        if payload.constraints.max_energy_above_hull is not None:
            search_kwargs["energy_above_hull"] = (
                0.0,
                float(payload.constraints.max_energy_above_hull),
            )
        if is_semiconductor_goal:
            search_kwargs["is_metal"] = False
            search_kwargs["num_elements"] = (2, 20)

        try:
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(**search_kwargs)
        except Exception:
            return None

        candidates: list[Candidate] = []
        for doc in docs:
            material_id = str(doc.material_id)
            formula = str(doc.formula_pretty)
            doc_elements = {str(e).lower() for e in getattr(doc, "elements", [])}
            elements = doc_elements or self._extract_elements(formula)
            mp_band_gap_ev = getattr(doc, "band_gap", None)
            nsites = getattr(doc, "nsites", None)
            if elements & banned:
                continue
            if required and not required.issubset(elements):
                continue
            if not self._passes_goal_semantics(
                goal=payload.research_goal,
                element_set=elements,
                mp_band_gap_ev=mp_band_gap_ev,
                min_band_gap_ev=payload.constraints.min_band_gap_ev,
            ):
                continue
            candidates.append(
                Candidate(
                    material_id=material_id,
                    formula=formula,
                    source="materials_project",
                    features={
                        "elements": [str(e) for e in getattr(doc, "elements", [])],
                        "mp_band_gap_ev": mp_band_gap_ev,
                        "nsites": nsites,
                        "structure": (
                            doc.structure.as_dict()
                            if getattr(doc, "structure", None) is not None
                            else None
                        ),
                    },
                )
            )

        # MP server-side ordering is not guaranteed by target property.
        # Rank locally by known MP band gap before truncating to request limit.
        candidates.sort(
            key=lambda c: (
                -(float(c.features.get("mp_band_gap_ev") or -1.0)),
                c.material_id,
            )
        )
        candidates = candidates[:limit]

        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(),
            output_summary={
                "candidate_count": len(candidates),
                "source": "materials_project",
                "fallback_used": False,
            },
        )
        return MPRetrieverOutput(candidates=candidates, provenance=provenance)

    def _retrieve_from_mock(self, payload: MPRetrieverInput, source: str) -> MPRetrieverOutput:
        mock_pool = [
            Candidate(
                material_id="mp-mock-001",
                formula="Al3Mg2",
                source="mock",
                features={"mp_band_gap_ev": 0.1, "nsites": 20},
            ),
            Candidate(
                material_id="mp-mock-002",
                formula="CuZn",
                source="mock",
                features={"mp_band_gap_ev": 0.0, "nsites": 2},
            ),
            Candidate(
                material_id="mp-mock-003",
                formula="AlN",
                source="mock",
                features={"mp_band_gap_ev": 5.9, "nsites": 8},
            ),
            Candidate(
                material_id="mp-mock-004",
                formula="SiC",
                source="mock",
                features={"mp_band_gap_ev": 2.4, "nsites": 8},
            ),
            Candidate(
                material_id="mp-mock-005",
                formula="Fe2VAl",
                source="mock",
                features={"mp_band_gap_ev": None, "nsites": 12},
            ),
            Candidate(
                material_id="mp-mock-006",
                formula="Ni3Al",
                source="mock",
                features={"mp_band_gap_ev": None, "nsites": 54},
            ),
            Candidate(
                material_id="mp-mock-007",
                formula="CoTi",
                source="mock",
                features={"mp_band_gap_ev": 1.0, "nsites": 4},
            ),
            Candidate(
                material_id="mp-mock-008",
                formula="O2",
                source="mock",
                features={"mp_band_gap_ev": 1.3, "nsites": 2},
            ),
        ]

        banned = {el.lower() for el in payload.constraints.banned_elements}
        required = {el.lower() for el in payload.constraints.required_elements}
        filtered = [
            c
            for c in mock_pool
            if not (self._extract_elements(c.formula) & banned)
            and required.issubset(self._extract_elements(c.formula))
            and self._passes_goal_semantics(
                goal=payload.research_goal,
                element_set=self._extract_elements(c.formula),
                mp_band_gap_ev=c.features.get("mp_band_gap_ev"),
                min_band_gap_ev=payload.constraints.min_band_gap_ev,
            )
        ][: payload.constraints.top_k * self.config.request_limit_multiplier]

        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(),
            output_summary={
                "candidate_count": len(filtered),
                "source": source,
                "fallback_used": True,
                "fallback_reason": (
                    "live_mp_returned_zero_candidates_after_server_and_client_filters"
                    if source == "mock_fallback_no_live_results"
                    else f"missing {self.config.api_key_env_var}, missing mp-api, or MP request error"
                ),
            },
        )
        return MPRetrieverOutput(candidates=filtered, provenance=provenance)

    @staticmethod
    def _extract_elements(formula: str) -> set[str]:
        # Matches element tokens like "Al", "Mg", "C", "Co" from formulas such as Al3Mg2.
        return {token.lower() for token in re.findall(r"[A-Z][a-z]?", formula)}

    @staticmethod
    def _passes_goal_semantics(
        goal: str,
        element_set: set[str],
        mp_band_gap_ev: float | None,
        min_band_gap_ev: float | None,
    ) -> bool:
        goal_lower = goal.lower()
        if "semiconductor" in goal_lower and len(element_set) < 2:
            return False
        if (
            min_band_gap_ev is not None
            and mp_band_gap_ev is not None
            and float(mp_band_gap_ev) < min_band_gap_ev
        ):
            return False
        return True
