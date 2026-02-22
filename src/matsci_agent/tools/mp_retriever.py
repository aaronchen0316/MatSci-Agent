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
        if self.config.use_live_if_available:
            live = self._retrieve_from_mp(payload)
            if live is not None:
                return live

        return self._retrieve_from_mock(payload, source="mock_fallback")

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

        try:
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    elements=payload.constraints.required_elements or None,
                    fields=["material_id", "formula_pretty", "elements"],
                    num_chunks=1,
                    chunk_size=max(20, limit * 3),
                )
        except Exception:
            return None

        candidates: list[Candidate] = []
        for doc in docs:
            material_id = str(doc.material_id)
            formula = str(doc.formula_pretty)
            doc_elements = {str(e).lower() for e in getattr(doc, "elements", [])}
            elements = doc_elements or self._extract_elements(formula)
            if elements & banned:
                continue
            if required and not required.issubset(elements):
                continue
            candidates.append(
                Candidate(
                    material_id=material_id,
                    formula=formula,
                    source="materials_project",
                    features={"elements": [str(e) for e in getattr(doc, "elements", [])]},
                )
            )
            if len(candidates) >= limit:
                break

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
            Candidate(material_id="mp-mock-001", formula="Al3Mg2", source="mock"),
            Candidate(material_id="mp-mock-002", formula="CuZn", source="mock"),
            Candidate(material_id="mp-mock-003", formula="AlN", source="mock"),
            Candidate(material_id="mp-mock-004", formula="SiC", source="mock"),
            Candidate(material_id="mp-mock-005", formula="Fe2VAl", source="mock"),
            Candidate(material_id="mp-mock-006", formula="Ni3Al", source="mock"),
            Candidate(material_id="mp-mock-007", formula="CoTi", source="mock"),
        ]

        banned = {el.lower() for el in payload.constraints.banned_elements}
        required = {el.lower() for el in payload.constraints.required_elements}
        filtered = [
            c
            for c in mock_pool
            if not (self._extract_elements(c.formula) & banned)
            and required.issubset(self._extract_elements(c.formula))
        ][: payload.constraints.top_k * self.config.request_limit_multiplier]

        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(),
            output_summary={
                "candidate_count": len(filtered),
                "source": source,
                "fallback_used": True,
                "fallback_reason": (
                    f"missing {self.config.api_key_env_var}, missing mp-api, or MP request error"
                ),
            },
        )
        return MPRetrieverOutput(candidates=filtered, provenance=provenance)

    @staticmethod
    def _extract_elements(formula: str) -> set[str]:
        # Matches element tokens like "Al", "Mg", "C", "Co" from formulas such as Al3Mg2.
        return {token.lower() for token in re.findall(r"[A-Z][a-z]?", formula)}
