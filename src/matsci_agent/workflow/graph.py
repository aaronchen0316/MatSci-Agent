from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from matsci_agent.agents.planner import ChemistryIntentAgent
from matsci_agent.agents.reporter import ResultsReportingAgent
from matsci_agent.config import settings
from matsci_agent.guardrails.capability import CapabilityGuardrail
from matsci_agent.observability.mlflow_logger import MLflowLogger
from matsci_agent.schemas import (
    Candidate,
    CapabilityAssessment,
    DiscoveryRequest,
    DiscoveryResponse,
    DiscoveryPlan,
    MPRetrieverInput,
    PredictedProperties,
    PropertyPredictorInput,
    PropertyPredictionRecord,
    RankedCandidate,
    ReportSummary,
    StabilityCheckerInput,
    ToolCallProvenance,
)
from matsci_agent.tools.mp_retriever import MPRetriever
from matsci_agent.tools.property_predictor import PropertyPredictor
from matsci_agent.tools.stability_checker import StabilityChecker
from matsci_agent.workflow.state import DiscoveryState


class DiscoveryWorkflow:
    def __init__(
        self,
        retriever: MPRetriever | None = None,
        predictor: PropertyPredictor | None = None,
        stability_checker: StabilityChecker | None = None,
        logger: MLflowLogger | None = None,
        intent_agent: ChemistryIntentAgent | None = None,
        capability_guardrail: CapabilityGuardrail | None = None,
        reporting_agent: ResultsReportingAgent | None = None,
    ) -> None:
        self.retriever = retriever or MPRetriever()
        self.predictor = predictor or PropertyPredictor()
        self.stability_checker = stability_checker or StabilityChecker()
        self.logger = logger or MLflowLogger(settings.mlflow_experiment)
        self.intent_agent = intent_agent or ChemistryIntentAgent()
        self.capability_guardrail = capability_guardrail or CapabilityGuardrail()
        self.reporting_agent = reporting_agent or ResultsReportingAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(DiscoveryState)
        workflow.add_node("plan_intent", self._plan_intent)
        workflow.add_node("assess_capability", self._assess_capability)
        workflow.add_node("retrieve", self._retrieve)
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
        workflow.add_edge("retrieve", "predict")
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

    def _predict(self, state: DiscoveryState) -> DiscoveryState:
        candidates = state.get("raw_candidates", [])
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
                not x.stability.is_stable,
                -x.predicted.band_gap_ev,
                x.stability.energy_above_hull,
            ),
        ):
            if (
                min_band_gap_ev is not None
                and rec.predicted.band_gap_ev < min_band_gap_ev
            ):
                continue
            score = rec.predicted.band_gap_ev - 100.0 * rec.stability.energy_above_hull
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

        stable_found = any(r.stability.is_stable for r in ranked)
        self.logger.log_step(
            "stability_checker",
            metrics={
                "stable_count": sum(1 for r in ranked if r.stability.is_stable),
                "ranked_count": len(ranked),
                "iteration": float(state["iteration"]),
            },
        )
        provenance = state.get("provenance", [])
        provenance.append(result.provenance.model_dump())
        return {
            "ranked_candidates": ranked[: state["constraints"].top_k],
            "stable_found": stable_found,
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
        report = self.reporting_agent.report(
            plan=plan,
            capability=capability,
            ranked_candidates=ranked,
            iteration=state.get("iteration", 1),
        )

        if not capability.supported:
            status = "unsupported"
        else:
            has_stable = any(r.stability.is_stable for r in ranked)
            status = "success" if has_stable else "partial" if ranked else "failed"

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

    def _route_after_stability(self, state: DiscoveryState) -> str:
        if state.get("stable_found"):
            return "done"
        if state["iteration"] >= state["max_iterations"]:
            return "done"
        return "retry"

    def run(self, request: DiscoveryRequest) -> DiscoveryResponse:
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
