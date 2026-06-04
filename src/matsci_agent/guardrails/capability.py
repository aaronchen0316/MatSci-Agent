from __future__ import annotations

from dataclasses import dataclass

from matsci_agent.schemas import CapabilityAssessment, DiscoveryPlan


@dataclass(frozen=True)
class TaskCapability:
    supported: bool
    reason_code: str | None = None
    reason_message: str | None = None
    closest_supported_mode: str | None = "band_gap_screening"
    suggested_next_action: str | None = None


TASK_REGISTRY: dict[str, TaskCapability] = {
    "band_gap_screening": TaskCapability(supported=True),
    "bulk_relaxation_only": TaskCapability(
        supported=False,
        reason_code="unsupported_relaxation_only",
        reason_message=(
            "Relaxation-only requests are unsupported because the current API is built "
            "for candidate screening and does not expose a standalone structure-relaxation workflow."
        ),
        suggested_next_action=(
            "Use a band-gap screening request and enable recalculation or relaxation as part of screening."
        ),
    ),
    "diffusivity_simulation": TaskCapability(
        supported=False,
        reason_code="unsupported_diffusivity",
        reason_message=(
            "Diffusivity is unsupported because the current system only implements bulk band-gap screening "
            "and does not include MD or transport-property simulation."
        ),
        suggested_next_action="Try a band-gap screening request over Materials Project candidates.",
    ),
    "molecular_dynamics": TaskCapability(
        supported=False,
        reason_code="unsupported_molecular_dynamics",
        reason_message=(
            "Molecular dynamics is unsupported because the current workflow only supports bounded screening "
            "and optional local relaxation, not long trajectory simulation."
        ),
        suggested_next_action="Try a bounded band-gap screening request instead of MD.",
    ),
    "transport_property_estimation": TaskCapability(
        supported=False,
        reason_code="unsupported_transport",
        reason_message=(
            "Transport-property estimation is unsupported because the current workflow only resolves band gap."
        ),
        suggested_next_action="Try a band-gap screening request instead of transport-property estimation.",
    ),
    "defect_property_workflow": TaskCapability(
        supported=False,
        reason_code="unsupported_defect_workflow",
        reason_message=(
            "Defect-property workflows are unsupported because the current system does not model defects or supercells."
        ),
        suggested_next_action="Try a bulk band-gap screening request against pristine MP entries.",
    ),
    "general_ab_initio_simulation": TaskCapability(
        supported=False,
        reason_code="unsupported_ab_initio",
        reason_message=(
            "General ab initio simulation is unsupported because the current codebase is a screening pipeline, "
            "not a generic electronic-structure workflow runner."
        ),
        suggested_next_action="Use the system for band-gap screening against Materials Project candidates.",
    ),
    "unknown_task": TaskCapability(
        supported=False,
        reason_code="unknown_task_class",
        reason_message=(
            "The request could not be mapped to a supported workflow in the current codebase."
        ),
        suggested_next_action="Rephrase the goal as a bulk materials band-gap screening request.",
    ),
}


class CapabilityGuardrail:
    def assess(self, plan: DiscoveryPlan) -> CapabilityAssessment:
        capability = TASK_REGISTRY.get(plan.task_class, TASK_REGISTRY["unknown_task"])
        if capability.supported:
            return CapabilityAssessment(supported=True)
        return CapabilityAssessment(
            supported=False,
            reason_code=capability.reason_code,
            reason_message=capability.reason_message,
            closest_supported_mode=capability.closest_supported_mode,
            suggested_next_action=capability.suggested_next_action,
        )
