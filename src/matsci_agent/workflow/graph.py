from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from matsci_agent.config import settings
from matsci_agent.observability.mlflow_logger import MLflowLogger
from matsci_agent.schemas import (
    Candidate,
    DiscoveryRequest,
    DiscoveryResponse,
    MPRetrieverInput,
    PropertyPredictorInput,
    PropertyPredictionRecord,
    RankedCandidate,
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
    ) -> None:
        self.retriever = retriever or MPRetriever()
        self.predictor = predictor or PropertyPredictor()
        self.stability_checker = stability_checker or StabilityChecker()
        self.logger = logger or MLflowLogger(settings.mlflow_experiment)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(DiscoveryState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("predict", self._predict)
        workflow.add_node("check_stability", self._check_stability)
        workflow.add_node("refine", self._refine)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "predict")
        workflow.add_edge("predict", "check_stability")
        workflow.add_conditional_edges(
            "check_stability",
            self._route_after_stability,
            {"done": END, "retry": "refine"},
        )
        workflow.add_edge("refine", "retrieve")
        return workflow.compile()

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
        payload = PropertyPredictorInput(
            candidates=[Candidate.model_validate(c) for c in candidates],
            goal=state["research_goal"],
            surrogate_mode=state["constraints"].surrogate_mode,
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
        min_tc = state["constraints"].min_thermal_conductivity

        ranked: list[RankedCandidate] = []
        for rec in sorted(
            result.records,
            key=lambda x: (
                not x.stability.is_stable,
                -x.predicted.thermal_conductivity,
                x.stability.energy_above_hull,
            ),
        ):
            if min_tc is not None and rec.predicted.thermal_conductivity < min_tc:
                continue
            score = rec.predicted.thermal_conductivity - 100.0 * rec.stability.energy_above_hull
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
        constraints.max_energy_above_hull = min(constraints.max_energy_above_hull + 0.03, 0.3)
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
            "iteration": iteration,
            "messages": messages,
        }

    def _route_after_stability(self, state: DiscoveryState) -> str:
        if state.get("stable_found"):
            return "done"
        if state["iteration"] >= state["max_iterations"]:
            return "done"
        return "retry"

    def run(self, request: DiscoveryRequest) -> DiscoveryResponse:
        initial_state: DiscoveryState = {
            "research_goal": request.research_goal,
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
        has_stable = any(r.stability.is_stable for r in ranked)
        status = "success" if has_stable else "partial" if ranked else "failed"

        return DiscoveryResponse(
            research_goal=request.research_goal,
            constraints=final_state.get("constraints", request.constraints),
            status=status,
            iterations=final_state.get("iteration", 1),
            candidates=ranked,
            provenance=[
                ToolCallProvenance.model_validate(p)
                for p in final_state.get("provenance", [])
            ],
            messages=final_state.get("messages", []),
        )
