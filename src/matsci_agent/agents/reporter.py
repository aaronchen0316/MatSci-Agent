from __future__ import annotations

from matsci_agent.schemas import CapabilityAssessment, DiscoveryPlan, RankedCandidate, ReportSummary


class ResultsReportingAgent:
    """Deterministic reporting layer for success and refusal paths."""

    def report(
        self,
        plan: DiscoveryPlan,
        capability: CapabilityAssessment,
        ranked_candidates: list[RankedCandidate],
        iteration: int,
    ) -> ReportSummary:
        if not capability.supported:
            reason = capability.reason_message or "Request is outside current system scope."
            suggestion = capability.suggested_next_action or "Try a supported band-gap screening request."
            return ReportSummary(
                scientific_summary=reason,
                execution_summary=f"Execution refused before retrieval. task_class={plan.task_class}.",
                caveats=[suggestion],
            )

        candidate_count = len(ranked_candidates)
        top_band_gap = (
            ranked_candidates[0].predicted_properties.band_gap_ev
            if ranked_candidates
            else None
        )
        summary = (
            f"Screened {candidate_count} candidates for task_class={plan.task_class}."
        )
        if top_band_gap is not None:
            summary = f"{summary} Top returned band gap: {top_band_gap:.3f} eV."
        return ReportSummary(
            scientific_summary=summary,
            execution_summary=(
                f"Completed in {iteration} iteration(s) with ranking_intent={plan.ranking_intent}."
            ),
            caveats=[],
        )
