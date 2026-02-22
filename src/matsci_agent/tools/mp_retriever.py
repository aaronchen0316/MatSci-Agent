from __future__ import annotations

from matsci_agent.schemas import (
    Candidate,
    MPRetrieverInput,
    MPRetrieverOutput,
    ToolCallProvenance,
)


class MPRetriever:
    """Materials Project retriever interface.

    TODO: Replace mock candidate generation with real MP API client call.
    """

    def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
        mock_pool = [
            Candidate(material_id="mp-mock-001", formula="Al3Mg2", source="mock"),
            Candidate(material_id="mp-mock-002", formula="CuZn", source="mock"),
            Candidate(material_id="mp-mock-003", formula="AlN", source="mock"),
            Candidate(material_id="mp-mock-004", formula="SiC", source="mock"),
            Candidate(material_id="mp-mock-005", formula="Fe2VAl", source="mock"),
            Candidate(material_id="mp-mock-006", formula="Ni3Al", source="mock"),
            Candidate(material_id="mp-mock-007", formula="CoTi", source="mock"),
        ]

        banned = set(el.lower() for el in payload.constraints.banned_elements)
        required = [el.lower() for el in payload.constraints.required_elements]
        filtered = [
            c
            for c in mock_pool
            if not any(el in c.formula.lower() for el in banned)
            and all(el in c.formula.lower() for el in required)
        ][: payload.constraints.top_k * 2]

        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(),
            output_summary={"candidate_count": len(filtered), "source": "mock"},
        )
        return MPRetrieverOutput(candidates=filtered, provenance=provenance)
