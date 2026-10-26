"""Offline Materials Project records for tests only."""

from matsci_agent.schemas import Candidate, MPRetrieverOutput, ToolCallProvenance
from matsci_agent.tools.mp_retriever import MPRetriever


_RECORDS = [
    ("001", "Al3Mg2", 0.1, 0.12, 20, True),
    ("002", "CuZn", 0.0, 0.06, 2, True),
    ("003", "AlN", 5.9, 0.01, 8, False),
    ("010", "AlN", 5.1, 0.02, 8, False),
    ("004", "SiC", 2.4, 0.02, 8, False),
    ("005", "Fe2VAl", None, None, 12, None),
    ("006", "Ni3Al", None, None, 54, None),
    ("007", "CoTi", 1.0, 0.03, 4, False),
    ("008", "O2", 1.3, 0.0, 2, False),
    ("009", "AcF3", 6.2, 0.02, 16, False),
]


class FixtureRetriever(MPRetriever):
    def retrieve(self, payload):
        limit = payload.limit_override or payload.constraints.top_k * self.config.request_limit_multiplier
        excluded = set(payload.exclude_material_ids)
        formulas = {target.normalized_formula for target in payload.search_space_targets}
        systems = {target.chemsys for target in payload.search_space_targets}
        candidates = []
        for material_id, formula, gap, hull, nsites, is_metal in _RECORDS:
            candidate = Candidate(
                material_id=f"mp-mock-{material_id}", formula=formula, source="mock",
                features={"mp_band_gap_ev": gap, "mp_energy_above_hull": hull, "nsites": nsites, "is_metal": is_metal},
            )
            if candidate.material_id in excluded or not self.passes_constraints(candidate, payload.constraints):
                continue
            elements = self._extract_elements(formula)
            chemsys = "-".join(sorted(element.capitalize() for element in elements))
            if (formulas or systems) and formula not in formulas and chemsys not in systems:
                continue
            candidates.append(candidate)
        candidates.sort(key=lambda candidate: self._candidate_rank_key(candidate, payload.research_goal))
        unique = []
        seen = set()
        for candidate in candidates:
            if candidate.formula not in seen:
                candidate.entry_count = sum(item.formula == candidate.formula for item in candidates)
                candidate.has_multiple_entries = candidate.entry_count > 1
                unique.append(candidate)
                seen.add(candidate.formula)
            if len(unique) >= limit:
                break
        return MPRetrieverOutput(
            candidates=unique,
            provenance=ToolCallProvenance(
                tool_name="test_retriever", input_payload=payload.model_dump(mode="json"),
                output_summary={"source": "test_fixture", "fallback_used": False, "candidate_count": len(unique)},
            ),
            reason_code=None if unique else "no_matches",
        )
