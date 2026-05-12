from __future__ import annotations

from matsci_agent.schemas import (
    StabilityCheckerInput,
    StabilityCheckerOutput,
    StabilityRecord,
    StabilityResult,
    ToolCallProvenance,
)


class StabilityChecker:
    """Stability checker with MP-only, honest-unknown semantics."""

    method = "mp_energy_above_hull_or_unknown"

    def run(self, payload: StabilityCheckerInput) -> StabilityCheckerOutput:
        records: list[StabilityRecord] = []
        mp_count = 0
        unknown_count = 0

        for record in payload.predictions:
            mp_energy_above_hull = record.candidate.features.get("mp_energy_above_hull")
            if isinstance(mp_energy_above_hull, (int, float)):
                energy_above_hull = round(float(mp_energy_above_hull), 6)
                source = "materials_project"
                used_relaxation = False
                method = "materials_project_energy_above_hull"
                stable = energy_above_hull <= payload.constraints.max_energy_above_hull
                mp_count += 1
            else:
                energy_above_hull = None
                source = "unknown"
                used_relaxation = False
                method = "stability_unknown_no_mp_hull"
                stable = None
                unknown_count += 1
            records.append(
                StabilityRecord(
                    candidate=record.candidate,
                    predicted=record.predicted,
                    stability=StabilityResult(
                        energy_above_hull=energy_above_hull,
                        is_stable=stable,
                        method=method,
                        source=source,
                        used_relaxation=used_relaxation,
                    ),
                )
            )

        provenance = ToolCallProvenance(
            tool_name="stability_checker",
            input_payload=payload.model_dump(),
            output_summary={
                "record_count": len(records),
                "stable_count": sum(1 for r in records if r.stability.is_stable is True),
                "method": self.method,
                "mp_count": mp_count,
                "unknown_count": unknown_count,
            },
        )
        return StabilityCheckerOutput(records=records, provenance=provenance)
