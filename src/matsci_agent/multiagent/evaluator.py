from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from matsci_agent.schemas import Candidate, DiscoveryFullResponse, DiscoveryRequest, MPRetrieverOutput, ToolCallProvenance
from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverConfig
from matsci_agent.workflow.graph import DiscoveryWorkflow

from matsci_agent.multiagent.schemas import (
    CompiledFilterEvidence,
    ConstraintViolationRecord,
    ConstraintViolationSummary,
    FailureStage,
    LiveEvalEvidence,
    LiveEvalInput,
    StageCounts,
)


class LiveOnlyMPRetriever(MPRetriever):
    """Evaluator retriever that never degrades into mock fallback."""

    def retrieve(self, payload) -> MPRetrieverOutput:
        live = self._retrieve_from_mp(payload)
        if live is not None:
            return live

        return MPRetrieverOutput(
            candidates=[],
            provenance=ToolCallProvenance(
                tool_name="mp_retriever",
                input_payload=payload.model_dump(mode="json"),
                output_summary={
                    "candidate_count": 0,
                    "source": "live_mp_unavailable",
                    "fallback_used": False,
                    "fallback_reason": f"missing {self.config.api_key_env_var}, missing mp-api, or MP request error",
                    "search_space_target_count": len(payload.search_space_targets),
                },
            ),
        )


@dataclass
class LiveRetrievalEvaluator:
    workflow_factory: Callable[[], DiscoveryWorkflow] | None = None
    retriever: MPRetriever | None = None

    def __post_init__(self) -> None:
        self.retriever = self.retriever or LiveOnlyMPRetriever(
            MPRetrieverConfig(use_live_if_available=True)
        )

    def evaluate(self, payload: LiveEvalInput) -> LiveEvalEvidence:
        if not payload.allow_live_mp:
            return self._blocked(payload.query, "live MP eval disabled")

        assert self.retriever is not None
        api_key_env = self.retriever.config.api_key_env_var
        if not os.getenv(api_key_env):
            return self._blocked(payload.query, f"missing {api_key_env}")

        workflow = self.workflow_factory() if self.workflow_factory is not None else DiscoveryWorkflow(retriever=self.retriever)
        response = workflow.run_full(DiscoveryRequest(research_goal=payload.query))
        return self._analyze_response(payload.query, response)

    def _blocked(self, query: str, reason: str) -> LiveEvalEvidence:
        return LiveEvalEvidence(status="blocked", query=query, blocked_reason=reason)

    def _analyze_response(self, query: str, response: DiscoveryFullResponse) -> LiveEvalEvidence:
        compiled_filters = self._compiled_filters(query, response)
        counts = StageCounts(
            raw_count=len(response.raw_candidates),
            filtered_count=len(response.filtered_candidates),
            ranked_count=len(response.candidates),
            search_space_target_count=len(response.search_space_targets),
            replenish_count=sum(1 for provenance in response.provenance if provenance.tool_name == "mp_retriever_replenish"),
        )
        violations = ConstraintViolationSummary(
            raw=self._violation_records(response.raw_candidates, "raw", query, response),
            filtered=self._violation_records(response.filtered_candidates, "filtered", query, response),
            ranked=self._violation_records([ranked.candidate for ranked in response.candidates], "ranked", query, response),
        )

        retriever_provenance = self._latest_provenance(response, "mp_retriever")
        retriever_source = None
        fallback_used = False
        if retriever_provenance is not None:
            retriever_source = retriever_provenance.output_summary.get("source")
            fallback_used = bool(retriever_provenance.output_summary.get("fallback_used"))

        if retriever_source in {"mock_fallback", "mock_fallback_no_live_results"} or fallback_used:
            return LiveEvalEvidence(
                status="blocked",
                query=query,
                compiled_filters=compiled_filters,
                result_counts=counts,
                constraint_violations=violations,
                messages=list(response.messages),
                provenance_summary=self._provenance_summary(response),
                blocked_reason="mock fallback used instead of live MP evidence",
            )

        if retriever_source == "live_mp_unavailable":
            return LiveEvalEvidence(
                status="blocked",
                query=query,
                compiled_filters=compiled_filters,
                result_counts=counts,
                constraint_violations=violations,
                messages=list(response.messages),
                provenance_summary=self._provenance_summary(response),
                blocked_reason="live MP request unavailable",
            )

        failed_stage = self._determine_failed_stage(response, compiled_filters, counts, violations, retriever_source)
        status = "pass" if failed_stage is None else "fail"
        return LiveEvalEvidence(
            status=status,
            query=query,
            failed_stage=failed_stage,
            compiled_filters=compiled_filters,
            result_counts=counts,
            constraint_violations=violations,
            messages=list(response.messages),
            provenance_summary=self._provenance_summary(response),
            real_source_used=retriever_source == "materials_project",
        )

    def _compiled_filters(self, query: str, response: DiscoveryFullResponse) -> CompiledFilterEvidence:
        assert self.retriever is not None
        effective = self.retriever._effective_filters(response.constraints, query)
        retriever_provenance = self._latest_provenance(response, "mp_retriever")
        mp_search_kwargs = {}
        mp_search_kwargs_sequence: list[dict[str, object]] = []
        if retriever_provenance is not None:
            mp_search_kwargs = dict(retriever_provenance.output_summary.get("search_kwargs") or {})
            mp_search_kwargs_sequence = list(retriever_provenance.output_summary.get("search_kwargs_sequence") or [])
        return CompiledFilterEvidence(
            effective_filters=effective.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            mp_search_kwargs=mp_search_kwargs,
            mp_search_kwargs_sequence=mp_search_kwargs_sequence,
        )

    def _violation_records(
        self,
        candidates: list[Candidate],
        stage: str,
        query: str,
        response: DiscoveryFullResponse,
    ) -> list[ConstraintViolationRecord]:
        records: list[ConstraintViolationRecord] = []
        assert self.retriever is not None
        effective = self.retriever._effective_filters(response.constraints, query)
        for candidate in candidates:
            violations = self._candidate_violations(candidate, effective)
            if violations:
                records.append(
                    ConstraintViolationRecord(
                        stage=stage,
                        material_id=candidate.material_id,
                        formula=candidate.formula,
                        violations=violations,
                    )
                )
        return records

    def _candidate_violations(self, candidate: Candidate, effective) -> list[str]:
        assert self.retriever is not None
        features = candidate.features
        elements = self.retriever._extract_elements(candidate.formula)
        violations: list[str] = []

        required_elements = {token.lower() for token in effective.elements}
        excluded_elements = {token.lower() for token in effective.exclude_elements}
        if required_elements and not required_elements.issubset(elements):
            violations.append("missing_required_elements")
        if excluded_elements & elements:
            violations.append("contains_excluded_elements")

        self._check_float_range(violations, "band_gap", features.get("mp_band_gap_ev"), effective.band_gap)
        self._check_float_range(violations, "energy_above_hull", features.get("mp_energy_above_hull"), effective.energy_above_hull)
        self._check_float_range(violations, "formation_energy", features.get("formation_energy"), effective.formation_energy)
        self._check_float_range(violations, "density", features.get("density"), effective.density)
        self._check_float_range(violations, "efermi", features.get("efermi"), effective.efermi)
        self._check_float_range(violations, "total_magnetization", features.get("total_magnetization"), effective.total_magnetization)
        self._check_float_range(violations, "volume", features.get("volume"), effective.volume)
        self._check_int_range(violations, "num_sites", features.get("nsites"), effective.num_sites)
        self._check_int_range(violations, "num_elements", len(elements), effective.num_elements)

        self._check_bool_match(violations, "is_metal", features.get("is_metal"), effective.is_metal)
        self._check_bool_match(violations, "theoretical", features.get("theoretical"), effective.theoretical)
        self._check_bool_match(violations, "deprecated", features.get("deprecated"), effective.deprecated)
        return violations

    @staticmethod
    def _check_float_range(violations: list[str], name: str, value: object, range_filter) -> None:
        if range_filter is None or value is None:
            return
        numeric = float(value)
        if range_filter.min is not None and numeric < range_filter.min:
            violations.append(f"{name}_below_min")
        if range_filter.max is not None and numeric > range_filter.max:
            violations.append(f"{name}_above_max")

    @staticmethod
    def _check_int_range(violations: list[str], name: str, value: object, range_filter) -> None:
        if range_filter is None or value is None:
            return
        numeric = int(value)
        if range_filter.min is not None and numeric < range_filter.min:
            violations.append(f"{name}_below_min")
        if range_filter.max is not None and numeric > range_filter.max:
            violations.append(f"{name}_above_max")

    @staticmethod
    def _check_bool_match(violations: list[str], name: str, value: object, expected: bool | None) -> None:
        if expected is None or value is None:
            return
        if bool(value) != expected:
            violations.append(f"{name}_mismatch")

    def _determine_failed_stage(
        self,
        response: DiscoveryFullResponse,
        compiled_filters: CompiledFilterEvidence,
        counts: StageCounts,
        violations: ConstraintViolationSummary,
        retriever_source: str | None,
    ) -> FailureStage | None:
        if response.capability_assessment is not None and not response.capability_assessment.supported:
            return "intent_parse"

        expander_provenance = self._latest_provenance(response, "search_space_expander")
        if expander_provenance is not None and expander_provenance.output_summary.get("failure_code"):
            return "search_space_expansion"

        if retriever_source == "materials_project" and not compiled_filters.mp_search_kwargs_sequence:
            return "mp_query_compilation"

        if retriever_source == "materials_project" and counts.raw_count == 0:
            return "mp_zero_results"

        raw_valid_count = counts.raw_count - len(violations.raw)
        filtered_valid_count = counts.filtered_count - len(violations.filtered)
        ranked_valid_count = counts.ranked_count - len(violations.ranked)

        if counts.raw_count > 0 and len(violations.raw) > 0 and raw_valid_count == 0:
            return "deterministic_filter"

        if raw_valid_count > 0 and filtered_valid_count == 0:
            return "llm_policy_filter"

        if filtered_valid_count > 0 and ranked_valid_count == 0:
            return "ranking"

        if response.status == "failed":
            return "unknown"

        return None

    @staticmethod
    def _latest_provenance(response: DiscoveryFullResponse, tool_name: str) -> ToolCallProvenance | None:
        matches = [provenance for provenance in response.provenance if provenance.tool_name == tool_name]
        return matches[-1] if matches else None

    @staticmethod
    def _provenance_summary(response: DiscoveryFullResponse) -> dict[str, object]:
        search_space_provenance = LiveRetrievalEvaluator._latest_provenance(response, "search_space_expander")
        retriever_provenance = LiveRetrievalEvaluator._latest_provenance(response, "mp_retriever")
        policy_provenance = LiveRetrievalEvaluator._latest_provenance(response, "policy_filter")
        return {
            "tool_names": [provenance.tool_name for provenance in response.provenance],
            "search_space_failure_code": (
                search_space_provenance.output_summary.get("failure_code")
                if search_space_provenance is not None
                else None
            ),
            "retriever_source": (
                retriever_provenance.output_summary.get("source")
                if retriever_provenance is not None
                else None
            ),
            "retriever_fallback_used": (
                retriever_provenance.output_summary.get("fallback_used")
                if retriever_provenance is not None
                else None
            ),
            "policy_failure_code": (
                policy_provenance.output_summary.get("failure_code")
                if policy_provenance is not None
                else None
            ),
        }
