from __future__ import annotations

import re
from collections.abc import Callable

from matsci_agent.config import settings
from matsci_agent.nlp.parser import merge_constraints, parse_goal_to_intent
from matsci_agent.schemas import (
    DiscoveryConstraints,
    DiscoveryPlan,
    ExecutionPolicy,
    ParsedDiscoveryIntent,
)


class ChemistryIntentAgent:
    """Converts a research goal into a typed execution plan.

    The only LLM call in this agent is the existing constraint parser. Task
    classification and execution-policy enrichment stay deterministic.
    """

    def __init__(
        self,
        parser_fn: Callable[[str], ParsedDiscoveryIntent | DiscoveryConstraints] | None = None,
    ) -> None:
        self.parser_fn = parser_fn or parse_goal_to_intent

    def plan(
        self,
        research_goal: str,
        base_constraints: DiscoveryConstraints,
        explicit_base_fields: set[str] | None = None,
    ) -> DiscoveryPlan:
        parsed = self.parser_fn(research_goal)
        if isinstance(parsed, DiscoveryConstraints):
            parsed_intent = ParsedDiscoveryIntent(constraints=parsed)
        else:
            parsed_intent = parsed
        requested_material_class = parsed_intent.requested_material_class
        resolved = merge_constraints(
            base_constraints,
            parsed_intent.constraints,
            explicit_base_fields=explicit_base_fields,
        )
        task_class = self._infer_task_class(research_goal, resolved)
        recalculate_top_n = self._extract_recalculate_top_n(research_goal)
        execution_policy = ExecutionPolicy(
            calculate_matgl=resolved.calculate_matgl,
            recalculate_top_n=(
                recalculate_top_n if resolved.calculate_matgl else None
            ),
            matgl_max_recalc_entries=settings.matgl_max_recalc_entries,
            matgl_max_atoms=settings.matgl_max_atoms,
            enable_relaxation=settings.matgl_enable_relaxation,
            relaxation_max_steps=settings.matgl_relaxation_max_steps,
        )
        return DiscoveryPlan(
            research_goal_raw=research_goal,
            task_class=task_class,
            parsed_constraints=resolved,
            application_intent=self._infer_application_intent(research_goal),
            source_universe="materials_project_entries",
            requested_material_class=requested_material_class,
            practicality_mode=self._infer_practicality_mode(research_goal),
            ranking_intent=self._infer_ranking_intent(research_goal, resolved),
            reporting_focus="compact_summary",
            execution_policy=execution_policy,
        )

    @staticmethod
    def _infer_task_class(
        goal: str,
        constraints: DiscoveryConstraints,
    ) -> str:
        text = goal.lower()
        if re.search(r"\bdiffus", text):
            return "diffusivity_simulation"
        if re.search(r"\bmolecular dynamics\b|\bmd\b|\btrajectory\b", text):
            return "molecular_dynamics"
        if re.search(r"\btransport\b|\bconductivit", text):
            return "transport_property_estimation"
        if re.search(r"\bdefect\b|\bvacanc", text):
            return "defect_property_workflow"
        if re.search(r"\bdft\b|\bab initio\b", text):
            return "general_ab_initio_simulation"
        if re.search(r"\brelax\b", text) and not (
            "band gap" in text
            or "semiconductor" in text
            or constraints.min_band_gap_ev is not None
        ):
            return "bulk_relaxation_only"
        if (
            "band gap" in text
            or "semiconductor" in text
            or constraints.min_band_gap_ev is not None
        ):
            return "band_gap_screening"
        return "unknown_task"

    @staticmethod
    def _infer_application_intent(goal: str) -> str:
        return "unknown"

    @staticmethod
    def _infer_practicality_mode(goal: str) -> str:
        return "unknown"

    @staticmethod
    def _infer_ranking_intent(goal: str, constraints: DiscoveryConstraints) -> str:
        text = goal.lower()
        if re.search(r"\brank.*band gap\b|\border.*band gap\b|\bsort.*band gap\b", text):
            return "band_gap_desc"
        if "band gap" in text or constraints.min_band_gap_ev is not None:
            return "band_gap_desc"
        return "default"

    @staticmethod
    def _extract_recalculate_top_n(goal: str) -> int | None:
        text = goal.lower()
        patterns = [
            r"\bredo\s+calculation\s+of\s+top\s+(\d{1,3})\b",
            r"\brecalculate\s+(?:the\s+)?top\s+(\d{1,3})\b",
            r"\brecalculate\s+top\s+(\d{1,3})\b",
            r"\brecompute\s+top\s+(\d{1,3})\b",
            r"\btop\s+(\d{1,3})\s+candidates?\s+.*\brecalculate\b",
            r"\btop\s+(\d{1,3})\s+candidates?\s+.*\bredo\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return max(1, min(100, int(match.group(1))))
        return None
