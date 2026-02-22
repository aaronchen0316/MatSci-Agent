from __future__ import annotations

from matsci_agent.schemas import (
    StabilityCheckerInput,
    StabilityCheckerOutput,
    StabilityRecord,
    StabilityResult,
    ToolCallProvenance,
)


class StabilityChecker:
    """Stability checker interface.

    TODO: Replace with CHGNet/M3GNet + phase diagram backed stability pipeline.
    """

    method = "mock_e_above_hull"

    def run(self, payload: StabilityCheckerInput) -> StabilityCheckerOutput:
        records: list[StabilityRecord] = []
        for record in payload.predictions:
            noise = (sum(ord(ch) for ch in record.candidate.formula) % 20) / 100
            energy_above_hull = round(noise, 4)
            stable = energy_above_hull <= payload.constraints.max_energy_above_hull
            records.append(
                StabilityRecord(
                    candidate=record.candidate,
                    predicted=record.predicted,
                    stability=StabilityResult(
                        energy_above_hull=energy_above_hull,
                        is_stable=stable,
                        method=self.method,
                    ),
                )
            )

        provenance = ToolCallProvenance(
            tool_name="stability_checker",
            input_payload=payload.model_dump(),
            output_summary={
                "record_count": len(records),
                "stable_count": sum(1 for r in records if r.stability.is_stable),
                "method": self.method,
            },
        )
        return StabilityCheckerOutput(records=records, provenance=provenance)
