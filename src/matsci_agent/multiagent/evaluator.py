from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from matsci_agent.schemas import Candidate, DiscoveryConstraints, DiscoveryFullResponse, DiscoveryRequest, MPRetrieverOutput, ToolCallProvenance
from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverConfig
from matsci_agent.workflow.graph import DiscoveryWorkflow

from matsci_agent.multiagent.schemas import (
    CandidateReviewSnapshot,
    CandidateReviewSnapshots,
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
        response = workflow.run_full(
            DiscoveryRequest(
                research_goal=payload.query,
                constraints=payload.constraints or DiscoveryConstraints(),
            )
        )
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
        snapshots = self._candidate_snapshots(response)

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
                candidate_snapshots=snapshots,
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
                candidate_snapshots=snapshots,
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
            candidate_snapshots=snapshots,
            real_source_used=retriever_source == "materials_project",
        )

    def _candidate_snapshots(self, response: DiscoveryFullResponse) -> CandidateReviewSnapshots:
        """Keep Critic evidence compact and limited to review-relevant MP fields."""

        records = {record.candidate.material_id: record for record in response.filter_records}
        return CandidateReviewSnapshots(
            raw=[self._candidate_snapshot(candidate, records.get(candidate.material_id)) for candidate in response.raw_candidates[:20]],
            filtered=[
                self._candidate_snapshot(candidate, records.get(candidate.material_id))
                for candidate in response.filtered_candidates[:20]
            ],
            ranked=[
                self._candidate_snapshot(
                    ranked.candidate,
                    records.get(ranked.candidate.material_id),
                    rank=ranked.rank,
                    score=ranked.score,
                    predicted_band_gap_ev=ranked.predicted_properties.band_gap_ev,
                    stability_energy_above_hull=ranked.stability.energy_above_hull,
                    is_stable=ranked.stability.is_stable,
                )
                for ranked in response.candidates
            ],
        )

    def _candidate_snapshot(
        self,
        candidate: Candidate,
        policy_record,
        *,
        rank: int | None = None,
        score: float | None = None,
        predicted_band_gap_ev: float | None = None,
        stability_energy_above_hull: float | None = None,
        is_stable: bool | None = None,
    ) -> CandidateReviewSnapshot:
        assert self.retriever is not None
        features = candidate.features
        raw_elements = features.get("elements")
        elements = sorted(str(element) for element in raw_elements) if isinstance(raw_elements, list) else sorted(
            self.retriever._extract_elements(candidate.formula)
        )
        return CandidateReviewSnapshot(
            material_id=candidate.material_id,
            formula=candidate.formula,
            elements=elements,
            mp_band_gap_ev=features.get("mp_band_gap_ev"),
            energy_above_hull=features.get("mp_energy_above_hull"),
            formation_energy=features.get("formation_energy"),
            density=features.get("density"),
            volume=features.get("volume"),
            num_sites=features.get("nsites"),
            is_metal=features.get("is_metal"),
            is_stable=is_stable if is_stable is not None else features.get("is_stable"),
            crystal_system=features.get("crystal_system"),
            spacegroup_number=features.get("spacegroup_number"),
            spacegroup_symbol=features.get("spacegroup_symbol"),
            policy_passed=policy_record.passed if policy_record is not None else None,
            policy_reasons=list(policy_record.reasons) if policy_record is not None else [],
            rank=rank,
            score=score,
            predicted_band_gap_ev=predicted_band_gap_ev,
            stability_energy_above_hull=stability_energy_above_hull,
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
        effective_payload = effective.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        return CompiledFilterEvidence(
            effective_filters=effective_payload,
            mp_search_kwargs=mp_search_kwargs,
            mp_search_kwargs_sequence=mp_search_kwargs_sequence,
            missing_filter_keys=self._missing_compiled_filter_keys(effective_payload, mp_search_kwargs_sequence),
        )

    @staticmethod
    def _missing_compiled_filter_keys(effective: dict[str, object], kwargs_sequence: list[dict[str, object]]) -> list[str]:
        if not kwargs_sequence:
            return []
        latest = kwargs_sequence[-1]
        required = {
            key: value
            for key, value in effective.items()
            if key
            in {
                "formula",
                "chemsys",
                "material_ids",
                "elements",
                "exclude_elements",
                "possible_species",
                "has_props",
                "is_metal",
                "is_stable",
                "is_gap_direct",
                "theoretical",
                "deprecated",
                "has_reconstructed",
                "crystal_system",
                "spacegroup_number",
                "spacegroup_symbol",
                "band_gap",
                "energy_above_hull",
                "formation_energy",
                "density",
                "efermi",
                "total_magnetization",
                "volume",
                "num_sites",
                "num_elements",
            }
        }
        missing: list[str] = []
        for key in required:
            if key not in latest:
                missing.append(key)
        return sorted(missing)

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
        self._check_string_match(violations, "crystal_system", features.get("crystal_system"), effective.crystal_system)
        self._check_scalar_or_list_match(violations, "spacegroup_number", features.get("spacegroup_number"), effective.spacegroup_number)
        self._check_scalar_or_list_match(violations, "spacegroup_symbol", features.get("spacegroup_symbol"), effective.spacegroup_symbol)
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

    @staticmethod
    def _check_string_match(violations: list[str], name: str, value: object, expected: str | None) -> None:
        if expected is None:
            return
        if value is None:
            violations.append(f"{name}_missing")
            return
        if str(value).lower().split(".")[-1] != expected.lower():
            violations.append(f"{name}_mismatch")

    @staticmethod
    def _check_scalar_or_list_match(violations: list[str], name: str, value: object, expected: object) -> None:
        if expected is None:
            return
        if value is None:
            violations.append(f"{name}_missing")
            return
        expected_values = expected if isinstance(expected, list) else [expected]
        if value not in expected_values:
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

        if retriever_source == "materials_project" and compiled_filters.missing_filter_keys:
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
