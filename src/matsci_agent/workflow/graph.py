from __future__ import annotations

from collections import defaultdict

from matsci_agent.agents.planner import ChemistryIntentAgent
from matsci_agent.agents.reporter import ResultsReportingAgent
from matsci_agent.agents.search_space_expander import SearchSpaceExpansionAgent, SearchSpaceExpansionError
from matsci_agent.config import settings
from matsci_agent.guardrails.capability import CapabilityGuardrail
from matsci_agent.observability.mlflow_logger import MLflowLogger
from matsci_agent.schemas import (
    Candidate,
    CapabilityAssessment,
    DiscoveryConstraints,
    DiscoveryFullResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    MPRetrieverInput,
    PolicyFilterInput,
    PolicyFilterRecord,
    PropertyPredictorInput,
    RankedCandidate,
    ReportSummary,
    SearchSpaceExpansionInput,
    SearchSpaceTarget,
    StabilityCheckerInput,
    ToolCallProvenance,
)
from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverError
from matsci_agent.tools.policy_filter import PolicyFilter, PolicyFilterError
from matsci_agent.tools.property_predictor import MatGLCalculationError, PropertyPredictor
from matsci_agent.tools.stability_checker import StabilityChecker


def _formula_key(formula: str) -> str:
    try:
        from pymatgen.core import Composition

        return Composition(formula).reduced_formula
    except Exception:
        return formula


def _number(value: object, missing: float) -> float:
    return float(value) if isinstance(value, (int, float)) else missing


class DiscoveryWorkflow:
    """One pass: plan, retrieve, screen entries, select formulas, annotate, report."""

    def __init__(
        self,
        retriever: MPRetriever | None = None,
        predictor: PropertyPredictor | None = None,
        policy_filter: PolicyFilter | None = None,
        stability_checker: StabilityChecker | None = None,
        logger: MLflowLogger | None = None,
        intent_agent: ChemistryIntentAgent | None = None,
        capability_guardrail: CapabilityGuardrail | None = None,
        reporting_agent: ResultsReportingAgent | None = None,
        search_expander: SearchSpaceExpansionAgent | None = None,
        enable_policy_filter: bool | None = None,
    ) -> None:
        self.retriever = retriever or MPRetriever()
        self.predictor = predictor or PropertyPredictor()
        self.policy_filter = policy_filter or PolicyFilter()
        self.stability_checker = stability_checker or StabilityChecker()
        self.logger = logger or MLflowLogger(settings.mlflow_experiment)
        self.intent_agent = intent_agent or ChemistryIntentAgent()
        self.capability_guardrail = capability_guardrail or CapabilityGuardrail()
        self.reporting_agent = reporting_agent or ResultsReportingAgent()
        self.search_expander = search_expander or SearchSpaceExpansionAgent()
        self.enable_policy_filter = True if enable_policy_filter is None else enable_policy_filter

    @staticmethod
    def _formula_sort_key(entries: list[Candidate], constraints: DiscoveryConstraints, band_gap_task: bool) -> tuple[float, float, str]:
        best_hull = min(_number(entry.features.get("mp_energy_above_hull"), 999.0) for entry in entries)
        best_gap = max(_number(entry.features.get("mp_band_gap_ev"), -1.0) for entry in entries)
        filters = constraints.mp_filters
        if band_gap_task or filters.band_gap is not None:
            target = -best_gap
        elif filters.formation_energy is not None:
            target = min(_number(entry.features.get("formation_energy"), 999.0) for entry in entries)
        elif filters.density is not None:
            values = [_number(entry.features.get("density"), -1.0) for entry in entries]
            target = -max(values) if filters.density.min is not None and filters.density.max is None else min(values)
        elif filters.volume is not None:
            values = [_number(entry.features.get("volume"), -1.0) for entry in entries]
            target = -max(values) if filters.volume.min is not None and filters.volume.max is None else min(values)
        else:
            target = best_hull
        return (target, best_hull, _formula_key(entries[0].formula))

    def _select_formulas(self, candidates: list[Candidate], constraints: DiscoveryConstraints, band_gap_task: bool) -> list[Candidate]:
        grouped: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[_formula_key(candidate.formula)].append(candidate)
        ordered = sorted(grouped.values(), key=lambda entries: self._formula_sort_key(entries, constraints, band_gap_task))
        selected: list[Candidate] = []
        for entries in ordered[: constraints.top_k]:
            entries.sort(key=lambda entry: (_number(entry.features.get("mp_energy_above_hull"), 999.0), -_number(entry.features.get("mp_band_gap_ev"), -1.0), entry.material_id))
            for entry in entries:
                entry.entry_count = len(entries)
                entry.has_multiple_entries = len(entries) > 1
                selected.append(entry)
        return selected

    def _screen(self, candidates: list[Candidate], plan, provenance: list[ToolCallProvenance]) -> tuple[list[Candidate], list[PolicyFilterRecord]]:
        accepted: list[Candidate] = []
        records: list[PolicyFilterRecord] = []
        for start in range(0, len(candidates), 10):
            payload = PolicyFilterInput(candidates=candidates[start:start + 10], discovery_plan=plan)
            result = self.policy_filter.run(payload) if self.enable_policy_filter else self.policy_filter.skip(payload, policy="disabled_pass_through")
            accepted.extend(result.filtered_candidates)
            records.extend(result.records)
            provenance.append(result.provenance)
        return accepted, records

    def _execute(self, request: DiscoveryRequest) -> dict:
        messages: list[str] = []
        provenance: list[ToolCallProvenance] = []
        result: dict = {
            "status": "failed", "reason_code": None, "messages": messages,
            "provenance": provenance, "candidates": [], "raw_candidates": [],
            "filtered_candidates": [], "filter_records": [], "search_space_targets": [],
            "known_stability_present": None, "iterations": 1,
        }
        with self.logger.run("discover"):
            plan = self.intent_agent.plan(
                request.research_goal,
                request.constraints,
                explicit_base_fields=set(request.constraints.model_fields_set),
            )
            constraints = plan.parsed_constraints
            result.update(discovery_plan=plan, constraints=constraints)
            capability = self.capability_guardrail.assess(plan)
            result["capability_assessment"] = capability
            if not capability.supported:
                result.update(status="unsupported", reason_code="unsupported_request")
                messages.append(capability.reason_message or "Unsupported request.")
                result["report_summary"] = self.reporting_agent.report(plan, capability, [], 1)
                return result

            try:
                expansion = self.search_expander.expand(
                    SearchSpaceExpansionInput(
                        research_goal=request.research_goal,
                        discovery_plan=plan,
                        target_count=min(constraints.top_k * 3, 30),
                    )
                )
            except SearchSpaceExpansionError as exc:
                result.update(reason_code="search_expansion_failed")
                messages.append(f"Search-space expansion failed: {exc.message}")
                provenance.append(ToolCallProvenance(tool_name="search_space_expander", input_payload={}, output_summary={"failure_code": exc.code}))
                return result
            result["search_space_targets"] = expansion.targets
            provenance.append(expansion.provenance)
            if not expansion.targets:
                result["reason_code"] = "search_expansion_failed"
                messages.append("Search-space expansion returned no targets.")
                return result

            try:
                retrieval = self.retriever.retrieve(
                    MPRetrieverInput(
                        research_goal=request.research_goal,
                        constraints=constraints,
                        search_space_targets=expansion.targets,
                    )
                )
            except MPRetrieverError as exc:
                result["reason_code"] = exc.code
                messages.append(str(exc))
                provenance.append(ToolCallProvenance(tool_name="mp_retriever", input_payload={}, output_summary={"failure_code": exc.code}))
                return result
            provenance.append(retrieval.provenance)
            result["raw_candidates"] = retrieval.candidates
            result["data_source"] = retrieval.provenance.output_summary.get("source", "unknown")
            if not retrieval.candidates:
                result.update(status="success", reason_code=retrieval.reason_code or "no_matches")
                messages.append("No Materials Project candidates matched the request." if result["reason_code"] == "no_matches" else "No candidates passed the requested MP constraints.")
                return result

            try:
                accepted, records = self._screen(retrieval.candidates, plan, provenance)
            except PolicyFilterError as exc:
                result["reason_code"] = "chemistry_screen_failed"
                messages.append(f"Chemistry filter failed: {exc.message}")
                provenance.append(ToolCallProvenance(tool_name="policy_filter", input_payload={}, output_summary={"failure_code": exc.code}))
                return result
            result["filter_records"] = records
            if not accepted:
                result.update(status="success", reason_code="chemistry_no_pass")
                messages.append("Chemistry screening rejected all retrieved structures.")
                return result

            accepted = [candidate for candidate in accepted if MPRetriever.passes_constraints(candidate, constraints)]
            result["filtered_candidates"] = accepted
            if not accepted:
                result.update(status="success", reason_code="final_constraint_no_pass")
                messages.append("No structures passed the final constraints.")
                return result

            selected = self._select_formulas(accepted, constraints, plan.task_class == "band_gap_screening")
            for candidate in selected:
                candidate.features["properties"] = {
                    key: value for key, value in candidate.features.items()
                    if key in {"formation_energy", "density", "volume", "efermi", "total_magnetization", "crystal_system", "spacegroup_symbol", "spacegroup_number", "nsites"} and value is not None
                }

            try:
                prediction = self.predictor.run(
                    PropertyPredictorInput(
                        candidates=selected,
                        goal=request.research_goal,
                        calculate_matgl=plan.execution_policy.calculate_matgl,
                        matgl_max_recalc_entries=plan.execution_policy.matgl_max_recalc_entries,
                        matgl_max_atoms=plan.execution_policy.matgl_max_atoms,
                        enable_relaxation=plan.execution_policy.enable_relaxation,
                        relaxation_max_steps=plan.execution_policy.relaxation_max_steps,
                    )
                )
            except MatGLCalculationError as exc:
                result["reason_code"] = exc.code
                messages.append(str(exc))
                return result
            provenance.append(prediction.provenance)
            stability = self.stability_checker.run(StabilityCheckerInput(predictions=prediction.predictions, constraints=constraints))
            provenance.append(stability.provenance)
            ranked = [
                RankedCandidate(
                    rank=index,
                    candidate=record.candidate,
                    predicted_properties=record.predicted,
                    stability=record.stability,
                    score=round((_number(record.candidate.features.get("mp_band_gap_ev"), 0.0) if plan.task_class == "band_gap_screening" else 0.0) - 100 * _number(record.stability.energy_above_hull, 0.0), 4),
                    provenance={"source": record.candidate.source},
                )
                for index, record in enumerate(stability.records, start=1)
            ]
            result["candidates"] = ranked
            result["known_stability_present"] = any(record.stability.is_stable is not None for record in ranked)
            if ranked and not result["known_stability_present"]:
                messages.append("Stability is unknown because MP energy above hull is missing.")
            failures = sum(record.predicted.error is not None for record in stability.records)
            if failures:
                result["status"] = "failed" if failures == len(stability.records) else "partial"
                result["reason_code"] = "matgl_calculation_failed"
                messages.append(f"MatGL failed for {failures} of {len(stability.records)} structures.")
            else:
                result["status"] = "success"
            result["report_summary"] = self.reporting_agent.report(plan, capability, ranked, 1)
            return result

    def run(self, request: DiscoveryRequest) -> DiscoveryResponse:
        return self._response(request, self._execute(request))

    def run_full(self, request: DiscoveryRequest) -> DiscoveryFullResponse:
        data = self._execute(request)
        base = self._response(request, data)
        return DiscoveryFullResponse(
            **base.model_dump(),
            raw_candidates=data["raw_candidates"],
            filtered_candidates=data["filtered_candidates"],
            filter_records=data["filter_records"],
            search_space_targets=data["search_space_targets"],
            known_stability_present=data["known_stability_present"],
        )

    @staticmethod
    def _response(request: DiscoveryRequest, data: dict) -> DiscoveryResponse:
        return DiscoveryResponse(
            research_goal=request.research_goal,
            constraints=data.get("constraints", request.constraints),
            status=data["status"],
            reason_code=data["reason_code"],
            data_source=data.get("data_source"),
            iterations=data["iterations"],
            candidates=data["candidates"],
            provenance=data["provenance"],
            messages=data["messages"],
            discovery_plan=data.get("discovery_plan"),
            capability_assessment=data.get("capability_assessment"),
            report_summary=data.get("report_summary"),
        )
