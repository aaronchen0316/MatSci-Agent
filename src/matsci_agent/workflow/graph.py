from __future__ import annotations

import os

from langgraph.graph import END, START, StateGraph

from matsci_agent.agents.planner import ChemistryIntentAgent
from matsci_agent.agents.reporter import ResultsReportingAgent
from matsci_agent.config import settings
from matsci_agent.guardrails.capability import CapabilityGuardrail
from matsci_agent.observability.mlflow_logger import MLflowLogger
from matsci_agent.schemas import (
    Candidate,
    CapabilityAssessment,
    DiscoveryFullResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    DiscoveryPlan,
    MPRetrieverInput,
    PolicyFilterInput,
    PolicyFilterRecord,
    PropertyPredictorInput,
    PropertyPredictionRecord,
    RankedCandidate,
    ReportSummary,
    StabilityCheckerInput,
    ToolCallProvenance,
)
from matsci_agent.tools.mp_retriever import MPRetriever
from matsci_agent.tools.policy_filter import PolicyFilter, PolicyFilterError
from matsci_agent.tools.property_predictor import PropertyPredictor
from matsci_agent.tools.stability_checker import StabilityChecker
from matsci_agent.workflow.state import DiscoveryState


class DiscoveryWorkflow:
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
        self.enable_policy_filter = (
            enable_policy_filter
            if enable_policy_filter is not None
            else os.getenv("MATSCI_ENABLE_POLICY_FILTER", "").lower() in {"1", "true", "yes", "on"}
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(DiscoveryState)
        workflow.add_node("plan_intent", self._plan_intent)
        workflow.add_node("assess_capability", self._assess_capability)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("policy_filter", self._policy_filter)
        workflow.add_node("predict", self._predict)
        workflow.add_node("check_stability", self._check_stability)
        workflow.add_node("refine", self._refine)
        workflow.add_node("report", self._report)
        workflow.add_edge(START, "plan_intent")
        workflow.add_edge("plan_intent", "assess_capability")
        workflow.add_conditional_edges(
            "assess_capability",
            self._route_after_capability,
            {"continue": "retrieve", "unsupported": "report"},
        )
        workflow.add_edge("retrieve", "policy_filter")
        workflow.add_conditional_edges(
            "policy_filter",
            self._route_after_policy_filter,
            {"continue": "predict", "stop": "report"},
        )
        workflow.add_edge("predict", "check_stability")
        workflow.add_conditional_edges(
            "check_stability",
            self._route_after_stability,
            {"done": "report", "retry": "refine"},
        )
        workflow.add_edge("refine", "retrieve")
        workflow.add_edge("report", END)
        return workflow.compile()

    def _plan_intent(self, state: DiscoveryState) -> DiscoveryState:
        request_constraints = state["request_constraints"]
        plan = self.intent_agent.plan(
            state["research_goal"],
            request_constraints,
            explicit_base_fields=set(request_constraints.model_fields_set),
        )
        self.logger.log_step(
            "intent_agent",
            metrics={"iteration": float(state["iteration"])},
            params={
                "task_class": plan.task_class,
                "calculate_matgl": plan.execution_policy.calculate_matgl,
                "recalculate_top_n": plan.execution_policy.recalculate_top_n or 0,
            },
        )
        provenance = state.get("provenance", [])
        provenance.append(
            ToolCallProvenance(
                tool_name="intent_agent",
                input_payload={
                    "research_goal": state["research_goal"],
                    "explicit_constraints": request_constraints.model_dump(),
                },
                output_summary={
                    "task_class": plan.task_class,
                    "application_intent": plan.application_intent,
                    "material_class": plan.material_class,
                    "ranking_intent": plan.ranking_intent,
                    "calculate_matgl": plan.execution_policy.calculate_matgl,
                    "recalculate_top_n": plan.execution_policy.recalculate_top_n,
                },
            ).model_dump()
        )
        return {
            "discovery_plan": plan,
            "constraints": plan.parsed_constraints,
            "provenance": provenance,
        }

    def _assess_capability(self, state: DiscoveryState) -> DiscoveryState:
        plan = state["discovery_plan"]
        capability = self.capability_guardrail.assess(plan)
        self.logger.log_step(
            "capability_guardrail",
            metrics={
                "supported": 1.0 if capability.supported else 0.0,
                "iteration": float(state["iteration"]),
            },
            params={"task_class": plan.task_class},
        )
        messages = state.get("messages", [])
        if not capability.supported and capability.reason_message:
            messages.append(capability.reason_message)
        provenance = state.get("provenance", [])
        provenance.append(
            ToolCallProvenance(
                tool_name="capability_guardrail",
                input_payload={"task_class": plan.task_class},
                output_summary=capability.model_dump(),
            ).model_dump()
        )
        return {
            "capability_assessment": capability,
            "messages": messages,
            "provenance": provenance,
        }

    def _retrieve(self, state: DiscoveryState) -> DiscoveryState:
        payload = MPRetrieverInput(
            research_goal=state["research_goal"], constraints=state["constraints"]
        )
        result = self.retriever.retrieve(payload)
        self.logger.log_step(
            "mp_retriever",
            metrics={
                "candidate_count": len(result.candidates),
                "iteration": float(state["iteration"]),
            },
        )
        provenance = state.get("provenance", [])
        provenance.append(result.provenance.model_dump())
        return {
            "raw_candidates": [c.model_dump() for c in result.candidates],
            "provenance": provenance,
        }

    def _policy_filter(self, state: DiscoveryState) -> DiscoveryState:
        all_candidates = [Candidate.model_validate(c) for c in state.get("raw_candidates", [])]
        plan = state["discovery_plan"]
        if not self.enable_policy_filter:
            result = self.policy_filter.skip(
                PolicyFilterInput(candidates=all_candidates, discovery_plan=plan),
                policy="disabled_mvp_pass_through",
            )
            provenance = state.get("provenance", [])
            provenance.append(result.provenance.model_dump())
            return {
                "filtered_candidates": [c.model_dump() for c in result.filtered_candidates],
                "filter_records": [r.model_dump() for r in result.records],
                "provenance": provenance,
                "filter_replenish_attempts": 0,
            }
        if plan.task_class != "band_gap_screening":
            result = self.policy_filter.run(
                PolicyFilterInput(candidates=all_candidates, discovery_plan=plan)
            )
            provenance = state.get("provenance", [])
            provenance.append(result.provenance.model_dump())
            return {
                "filtered_candidates": [c.model_dump() for c in result.filtered_candidates],
                "filter_records": [r.model_dump() for r in result.records],
                "provenance": provenance,
                "filter_replenish_attempts": 0,
            }

        provenance = state.get("provenance", [])
        messages = state.get("messages", [])
        selected = self._select_filter_candidates(all_candidates, limit=10)
        try:
            first_result = self.policy_filter.run(
                PolicyFilterInput(candidates=selected, discovery_plan=plan)
            )
        except PolicyFilterError as exc:
            return self._policy_filter_failed_state(
                state,
                batch_size=len(selected),
                replenish_attempts=0,
                error=exc,
            )

        accepted = list(first_result.filtered_candidates)
        records = list(first_result.records)
        seen_ids = {candidate.material_id for candidate in selected}
        replenish_attempts = 0

        if len(accepted) < state["constraints"].top_k:
            replenish_attempts = 1
            deficit = state["constraints"].top_k - len(accepted)
            refill = self.retriever.retrieve(
                MPRetrieverInput(
                    research_goal=state["research_goal"],
                    constraints=state["constraints"],
                    exclude_material_ids=sorted(seen_ids),
                    limit_override=max(10, deficit),
                )
            )
            refill_candidates = [Candidate.model_validate(c.model_dump()) for c in refill.candidates]
            refill_selected = self._select_filter_candidates(refill_candidates, limit=10)
            if refill_selected:
                try:
                    second_result = self.policy_filter.run(
                        PolicyFilterInput(candidates=refill_selected, discovery_plan=plan)
                    )
                except PolicyFilterError as exc:
                    return self._policy_filter_failed_state(
                        state,
                        batch_size=len(refill_selected),
                        replenish_attempts=1,
                        error=exc,
                    )
                accepted.extend(second_result.filtered_candidates)
                records.extend(second_result.records)
                provenance.append(
                    ToolCallProvenance(
                        tool_name="mp_retriever_replenish",
                        input_payload={
                            "research_goal": state["research_goal"],
                            "exclude_material_ids": sorted(seen_ids),
                            "limit_override": max(10, deficit),
                        },
                        output_summary={
                            "candidate_count": len(refill_candidates),
                        },
                    ).model_dump()
                )
                provenance.append(second_result.provenance.model_dump())

        self.logger.log_step(
            "policy_filter",
            metrics={
                "input_count": float(len(selected)),
                "filtered_count": float(len(accepted)),
                "excluded_count": float(len(records) - len(accepted)),
                "iteration": float(state["iteration"]),
            },
            params={"policy": first_result.provenance.output_summary.get("policy", "unknown")},
        )
        provenance.append(first_result.provenance.model_dump())
        excluded_count = len(records) - len(accepted)
        if excluded_count:
            messages.append(
                f"Policy filter excluded {excluded_count} candidate(s) using {first_result.provenance.output_summary.get('policy', 'unknown')}."
            )
        if replenish_attempts:
            messages.append("Policy filter ran one replenish pass to fill requested candidate count.")
        next_state: DiscoveryState = {
            "filtered_candidates": [c.model_dump() for c in accepted],
            "filter_records": [r.model_dump() for r in records],
            "messages": messages,
            "provenance": provenance,
            "filter_replenish_attempts": replenish_attempts,
        }
        if not accepted:
            messages.append("Chemistry filter kept zero candidates for this request.")
            next_state["status"] = "partial"
        return next_state

    def _predict(self, state: DiscoveryState) -> DiscoveryState:
        candidates = state.get("filtered_candidates", state.get("raw_candidates", []))
        plan = state.get("discovery_plan")
        if isinstance(plan, DiscoveryPlan):
            policy = plan.execution_policy
        else:
            policy = None
        payload = PropertyPredictorInput(
            candidates=[Candidate.model_validate(c) for c in candidates],
            goal=state["research_goal"],
            calculate_matgl=(
                policy.calculate_matgl if policy is not None else state["constraints"].calculate_matgl
            ),
            recalculate_top_n=(policy.recalculate_top_n if policy is not None else None),
            matgl_max_recalc_entries=(
                policy.matgl_max_recalc_entries
                if policy is not None
                else settings.matgl_max_recalc_entries
            ),
            matgl_max_atoms=(
                policy.matgl_max_atoms if policy is not None else settings.matgl_max_atoms
            ),
            enable_relaxation=(
                policy.enable_relaxation
                if policy is not None
                else settings.matgl_enable_relaxation
            ),
            relaxation_max_steps=(
                policy.relaxation_max_steps
                if policy is not None
                else settings.matgl_relaxation_max_steps
            ),
        )
        result = self.predictor.run(payload)
        self.logger.log_step(
            "property_predictor",
            metrics={
                "prediction_count": len(result.predictions),
                "iteration": float(state["iteration"]),
            },
            params={
                "backend": result.provenance.output_summary.get("backend", "unknown"),
            },
        )
        provenance = state.get("provenance", [])
        provenance.append(result.provenance.model_dump())
        return {
            "predictions": [r.model_dump() for r in result.predictions],
            "provenance": provenance,
        }

    def _check_stability(self, state: DiscoveryState) -> DiscoveryState:
        payload = StabilityCheckerInput(
            predictions=[
                PropertyPredictionRecord.model_validate(record)
                for record in state.get("predictions", [])
            ],
            constraints=state["constraints"],
        )
        result = self.stability_checker.run(payload)
        min_band_gap_ev = state["constraints"].min_band_gap_ev

        ranked: list[RankedCandidate] = []
        for rec in sorted(
            result.records,
            key=lambda x: (
                -x.predicted.band_gap_ev,
                0 if x.stability.is_stable is True else 1 if x.stability.is_stable is None else 2,
                (
                    x.stability.energy_above_hull
                    if x.stability.energy_above_hull is not None
                    else 999.0
                ),
            ),
        ):
            if (
                min_band_gap_ev is not None
                and rec.predicted.band_gap_ev < min_band_gap_ev
            ):
                continue
            hull_penalty = (
                100.0 * rec.stability.energy_above_hull
                if rec.stability.energy_above_hull is not None
                else 0.0
            )
            score = rec.predicted.band_gap_ev - hull_penalty
            ranked.append(
                RankedCandidate(
                    rank=len(ranked) + 1,
                    candidate=rec.candidate,
                    predicted_properties=rec.predicted,
                    stability=rec.stability,
                    score=round(score, 4),
                    provenance={"iteration": state["iteration"]},
                )
            )

        stable_found = any(r.stability.is_stable is True for r in ranked)
        known_stability_present = any(r.stability.is_stable is not None for r in ranked)
        messages = state.get("messages", [])
        if ranked and not known_stability_present:
            messages.append(
                "Stability is unknown for returned candidates because MP energy_above_hull data is missing."
            )
        self.logger.log_step(
            "stability_checker",
            metrics={
                "stable_count": sum(1 for r in ranked if r.stability.is_stable is True),
                "ranked_count": len(ranked),
                "iteration": float(state["iteration"]),
            },
        )
        provenance = state.get("provenance", [])
        provenance.append(result.provenance.model_dump())
        return {
            "ranked_candidates": ranked[: state["constraints"].top_k],
            "stable_found": stable_found,
            "messages": messages,
            "known_stability_present": known_stability_present,
            "provenance": provenance,
        }

    def _refine(self, state: DiscoveryState) -> DiscoveryState:
        constraints = state["constraints"].model_copy(deep=True)
        constraints.max_energy_above_hull = min(
            constraints.max_energy_above_hull + 0.03, 0.3
        )
        if isinstance(state.get("discovery_plan"), DiscoveryPlan):
            plan = state["discovery_plan"].model_copy(deep=True)
            plan.parsed_constraints = constraints
        else:
            plan = None
        iteration = state["iteration"] + 1
        messages = state.get("messages", [])
        messages.append(
            f"No stable candidates at iteration {state['iteration']}; relaxed max_energy_above_hull to {constraints.max_energy_above_hull:.2f}."
        )
        self.logger.log_step(
            "refine",
            metrics={
                "iteration": float(iteration),
                "max_energy_above_hull": constraints.max_energy_above_hull,
            },
        )
        return {
            "constraints": constraints,
            "discovery_plan": plan,
            "iteration": iteration,
            "messages": messages,
        }

    def _report(self, state: DiscoveryState) -> DiscoveryState:
        plan = state.get("discovery_plan") or DiscoveryPlan(
            research_goal_raw=state["research_goal"],
            parsed_constraints=state["constraints"],
        )
        capability = state.get("capability_assessment") or CapabilityAssessment(
            supported=True
        )
        ranked = state.get("ranked_candidates", [])
        preset_status = state.get("status")
        if not capability.supported:
            report = self.reporting_agent.report(
                plan=plan,
                capability=capability,
                ranked_candidates=ranked,
                iteration=state.get("iteration", 1),
            )
            status = "unsupported"
        elif preset_status in {"failed", "partial"} and not ranked:
            report = self._make_preexecution_report(state, preset_status)
            status = preset_status
        else:
            report = self.reporting_agent.report(
                plan=plan,
                capability=capability,
                ranked_candidates=ranked,
                iteration=state.get("iteration", 1),
            )
            status = "success" if ranked else "failed"

        messages = state.get("messages", [])
        if not capability.supported:
            if report.caveats:
                messages.extend(report.caveats)
        else:
            messages.append(report.execution_summary)

        provenance = state.get("provenance", [])
        provenance.append(
            ToolCallProvenance(
                tool_name="reporting_agent",
                input_payload={
                    "task_class": plan.task_class,
                    "candidate_count": len(ranked),
                    "supported": capability.supported,
                },
                output_summary={
                    "status": status,
                    "caveat_count": len(report.caveats),
                },
            ).model_dump()
        )
        return {
            "report_summary": report,
            "status": status,
            "messages": messages,
            "provenance": provenance,
        }

    def _route_after_capability(self, state: DiscoveryState) -> str:
        capability = state.get("capability_assessment")
        if capability is not None and not capability.supported:
            return "unsupported"
        return "continue"

    def _route_after_policy_filter(self, state: DiscoveryState) -> str:
        if state.get("status") in {"failed", "partial"} and not state.get("filtered_candidates"):
            return "stop"
        return "continue"

    def _route_after_stability(self, state: DiscoveryState) -> str:
        return "done"

    def run(self, request: DiscoveryRequest) -> DiscoveryResponse:
        final_state = self._invoke(request)
        return self._build_response(request, final_state)

    def run_full(self, request: DiscoveryRequest) -> DiscoveryFullResponse:
        final_state = self._invoke(request)
        ranked = final_state.get("ranked_candidates", [])
        return DiscoveryFullResponse(
            research_goal=request.research_goal,
            constraints=final_state.get("constraints", request.constraints),
            status=final_state.get("status", "failed"),
            iterations=final_state.get("iteration", 1),
            candidates=ranked,
            provenance=[
                ToolCallProvenance.model_validate(p)
                for p in final_state.get("provenance", [])
            ],
            messages=final_state.get("messages", []),
            discovery_plan=final_state.get("discovery_plan"),
            capability_assessment=final_state.get("capability_assessment"),
            report_summary=final_state.get("report_summary"),
            raw_candidates=[
                Candidate.model_validate(c)
                for c in final_state.get("raw_candidates", [])
            ],
            filtered_candidates=[
                Candidate.model_validate(c)
                for c in final_state.get("filtered_candidates", [])
            ],
            filter_records=[
                PolicyFilterRecord.model_validate(r)
                for r in final_state.get("filter_records", [])
            ],
            known_stability_present=final_state.get("known_stability_present"),
        )

    def _invoke(self, request: DiscoveryRequest) -> DiscoveryState:
        initial_state: DiscoveryState = {
            "research_goal": request.research_goal,
            "request_constraints": request.constraints,
            "constraints": request.constraints,
            "iteration": 1,
            "max_iterations": settings.max_iterations,
            "messages": [],
            "provenance": [],
            "status": "failed",
        }

        with self.logger.run("discover"):
            final_state = self.graph.invoke(initial_state)
        return final_state

    def _build_response(
        self,
        request: DiscoveryRequest,
        final_state: DiscoveryState,
    ) -> DiscoveryResponse:
        ranked = final_state.get("ranked_candidates", [])
        return DiscoveryResponse(
            research_goal=request.research_goal,
            constraints=final_state.get("constraints", request.constraints),
            status=final_state.get("status", "failed"),
            iterations=final_state.get("iteration", 1),
            candidates=ranked,
            provenance=[
                ToolCallProvenance.model_validate(p)
                for p in final_state.get("provenance", [])
            ],
            messages=final_state.get("messages", []),
            discovery_plan=final_state.get("discovery_plan"),
            capability_assessment=final_state.get("capability_assessment"),
            report_summary=final_state.get("report_summary"),
        )

    @staticmethod
    def _cheap_candidate_sort_key(candidate: Candidate) -> tuple[float, float, int, str]:
        mp_gap = candidate.features.get("mp_band_gap_ev")
        mp_hull = candidate.features.get("mp_energy_above_hull")
        completeness = sum(
            1
            for key in ("elements", "mp_band_gap_ev", "mp_energy_above_hull", "nsites", "structure")
            if candidate.features.get(key) is not None
        )
        return (
            -(float(mp_gap) if isinstance(mp_gap, (int, float)) else -1.0),
            float(mp_hull) if isinstance(mp_hull, (int, float)) else 999.0,
            -completeness,
            candidate.material_id,
        )

    def _select_filter_candidates(self, candidates: list[Candidate], limit: int) -> list[Candidate]:
        return sorted(candidates, key=self._cheap_candidate_sort_key)[:limit]

    def _policy_filter_failed_state(
        self,
        state: DiscoveryState,
        batch_size: int,
        replenish_attempts: int,
        error: PolicyFilterError,
    ) -> DiscoveryState:
        messages = state.get("messages", [])
        messages.append(f"Chemistry filter failed: {error.message}")
        provenance = state.get("provenance", [])
        provenance.append(
            ToolCallProvenance(
                tool_name="policy_filter",
                input_payload={
                    "task_class": state["discovery_plan"].task_class,
                    "batch_size": batch_size,
                },
                output_summary={
                    "provider": self.policy_filter.provider,
                    "model": self.policy_filter.model,
                    "failure_code": error.code,
                    "raw_response_preview": error.raw_response_preview,
                    "replenish_attempts": replenish_attempts,
                },
            ).model_dump()
        )
        self.logger.log_step(
            "policy_filter",
            metrics={
                "input_count": float(batch_size),
                "filtered_count": 0.0,
                "excluded_count": 0.0,
                "iteration": float(state["iteration"]),
            },
            params={
                "policy": "llm",
                "failure_code": error.code,
            },
        )
        return {
            "filtered_candidates": [],
            "filter_records": [],
            "messages": messages,
            "provenance": provenance,
            "status": "failed",
            "filter_replenish_attempts": replenish_attempts,
        }

    @staticmethod
    def _make_preexecution_report(state: DiscoveryState, status: str) -> ReportSummary:
        if status == "failed":
            message = (
                state.get("messages", [])[-1]
                if state.get("messages")
                else "Execution failed before prediction."
            )
            return ReportSummary(
                scientific_summary=message,
                execution_summary="Execution stopped before prediction.",
                caveats=[],
            )
        return ReportSummary(
            scientific_summary="Chemistry filter kept zero candidates for this request.",
            execution_summary="Execution stopped after filtering because no candidates remained.",
            caveats=[],
        )
