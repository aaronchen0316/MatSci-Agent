from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from matsci_agent.nlp.parser import LLMConstraintParser
from matsci_agent.schemas import (
    Candidate,
    PolicyFilterBatchResponse,
    PolicyFilterDecisionPayload,
    PolicyFilterInput,
    PolicyFilterOutput,
    PolicyFilterRecord,
    ToolCallProvenance,
)
from matsci_agent.tools.mp_retriever import MPRetriever

_MAX_REASONS = 3
_MAX_REASON_CHARS = 80
_MAX_RESPONSE_TOKENS = 600
_TIMEOUT_SECS = 20.0


class PolicyFilterError(RuntimeError):
    def __init__(self, code: str, message: str, raw_response_preview: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.raw_response_preview = raw_response_preview[:500]


class PolicyFilter:
    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        inference_fn: Callable[[dict[str, Any]], dict[str, Any] | str] | None = None,
    ) -> None:
        self.provider = (provider or os.getenv("MATSCI_NLP_PROVIDER", "auto")).lower()
        self.model = model or os.getenv("MATSCI_NLP_MODEL", "")
        self.inference_fn = inference_fn
        self.last_response_preview: str = ""

    def skip(self, payload: PolicyFilterInput, policy: str) -> PolicyFilterOutput:
        return self._pass_through(payload, policy=policy)

    def run(self, payload: PolicyFilterInput) -> PolicyFilterOutput:
        policy = self._policy_name(payload)
        if payload.discovery_plan.task_class != "band_gap_screening":
            return self._pass_through(payload, policy="skipped_non_band_gap_task")

        raw = self._call_llm(payload, policy)
        batch = self._validate_batch_response(raw, payload, policy)

        records: list[PolicyFilterRecord] = []
        filtered: list[Candidate] = []
        by_id = {candidate.material_id: candidate for candidate in payload.candidates}

        for decision in batch.decisions:
            candidate = by_id[decision.material_id]
            candidate.features["filter_passed"] = decision.keep
            candidate.features["filter_reasons"] = list(decision.reasons)
            candidate.features["filter_policy"] = policy
            candidate.features["filter_source"] = "llm"
            record = PolicyFilterRecord(
                candidate=candidate,
                passed=decision.keep,
                reasons=list(decision.reasons),
                policy=policy,
            )
            records.append(record)
            if decision.keep:
                filtered.append(candidate)

        provenance = ToolCallProvenance(
            tool_name="policy_filter",
            input_payload=self._prompt_payload(payload, policy),
            output_summary={
                "policy": policy,
                "provider": self.provider,
                "model": self.model,
                "batch_size": len(payload.candidates),
                "filtered_count": len(filtered),
                "excluded_count": len(payload.candidates) - len(filtered),
                "raw_response_preview": self.last_response_preview,
            },
        )
        return PolicyFilterOutput(
            filtered_candidates=filtered,
            records=records,
            provenance=provenance,
        )

    def _pass_through(self, payload: PolicyFilterInput, policy: str) -> PolicyFilterOutput:
        records: list[PolicyFilterRecord] = []
        for candidate in payload.candidates:
            candidate.features["filter_passed"] = True
            candidate.features["filter_reasons"] = []
            candidate.features["filter_policy"] = policy
            candidate.features["filter_source"] = "llm_skipped"
            records.append(
                PolicyFilterRecord(
                    candidate=candidate,
                    passed=True,
                    reasons=[],
                    policy=policy,
                )
            )
        provenance = ToolCallProvenance(
            tool_name="policy_filter",
            input_payload=self._prompt_payload(payload, policy),
            output_summary={
                "policy": policy,
                "provider": self.provider,
                "model": self.model,
                "batch_size": len(payload.candidates),
                "filtered_count": len(payload.candidates),
                "excluded_count": 0,
                "raw_response_preview": "",
            },
        )
        return PolicyFilterOutput(
            filtered_candidates=list(payload.candidates),
            records=records,
            provenance=provenance,
        )

    def _call_llm(self, payload: PolicyFilterInput, policy: str) -> dict[str, Any]:
        self.last_response_preview = ""
        if self.inference_fn is not None:
            result = self.inference_fn(self._prompt_payload(payload, policy))
            if isinstance(result, str):
                self.last_response_preview = result[:500]
                return self._parse_response_json(result)
            if isinstance(result, dict):
                self.last_response_preview = json.dumps(result)[:500]
                return result
            raise PolicyFilterError(
                "policy_filter_invalid_json",
                "LLM chemistry filter returned unsupported response type.",
            )

        provider = self.provider
        if provider == "auto":
            if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
                return self._call_anthropic(payload, policy)
            if os.getenv("OPENAI_API_KEY"):
                return self._call_openai(payload, policy)
            raise PolicyFilterError(
                "policy_filter_llm_request_failed",
                "Chemistry filter failed because no LLM provider credentials are available.",
            )
        if provider == "anthropic":
            return self._call_anthropic(payload, policy)
        if provider == "openai":
            return self._call_openai(payload, policy)
        raise PolicyFilterError(
            "policy_filter_llm_request_failed",
            f"Chemistry filter failed because provider '{provider}' is unsupported.",
        )

    def _call_anthropic(self, payload: PolicyFilterInput, policy: str) -> dict[str, Any]:
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise PolicyFilterError(
                "policy_filter_llm_request_failed",
                "Chemistry filter failed because Anthropic credentials are missing.",
            )
        try:
            from anthropic import Anthropic
        except Exception as exc:
            raise PolicyFilterError(
                "policy_filter_llm_request_failed",
                f"Chemistry filter failed because Anthropic import failed: {exc}",
            ) from exc

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": _TIMEOUT_SECS}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = Anthropic(**client_kwargs)
        model = self.model or os.getenv(
            "MATSCI_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001-thinking"
        )
        try:
            response = client.messages.create(
                model=model,
                max_tokens=_MAX_RESPONSE_TOKENS,
                temperature=0,
                system=self._system_prompt(policy),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(self._prompt_payload(payload, policy)),
                    }
                ],
            )
            parts = []
            for block in getattr(response, "content", []):
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            content = "\n".join(parts) if parts else "{}"
            self.last_response_preview = content[:500]
            return self._parse_response_json(content)
        except Exception as exc:
            code = "policy_filter_timeout" if self._is_timeout_exc(exc) else "policy_filter_llm_request_failed"
            raise PolicyFilterError(
                code,
                f"Chemistry filter failed during Anthropic request: {exc}",
                self.last_response_preview,
            ) from exc

    def _call_openai(self, payload: PolicyFilterInput, policy: str) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise PolicyFilterError(
                "policy_filter_llm_request_failed",
                "Chemistry filter failed because OpenAI credentials are missing.",
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            raise PolicyFilterError(
                "policy_filter_llm_request_failed",
                f"Chemistry filter failed because OpenAI import failed: {exc}",
            ) from exc

        client = OpenAI(api_key=api_key, timeout=_TIMEOUT_SECS)
        model = self.model or os.getenv("MATSCI_OPENAI_MODEL", "gpt-4.1-mini")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=_MAX_RESPONSE_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt(policy)},
                    {
                        "role": "user",
                        "content": json.dumps(self._prompt_payload(payload, policy)),
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            self.last_response_preview = content[:500]
            return self._parse_response_json(content)
        except Exception as exc:
            code = "policy_filter_timeout" if self._is_timeout_exc(exc) else "policy_filter_llm_request_failed"
            raise PolicyFilterError(
                code,
                f"Chemistry filter failed during OpenAI request: {exc}",
                self.last_response_preview,
            ) from exc

    @staticmethod
    def _parse_response_json(content: str) -> dict[str, Any]:
        parsed = LLMConstraintParser._safe_json_dict(content)
        if not parsed:
            raise PolicyFilterError(
                "policy_filter_invalid_json",
                "Chemistry filter returned invalid JSON.",
                content[:500],
            )
        return parsed

    def _validate_batch_response(
        self,
        raw: dict[str, Any],
        payload: PolicyFilterInput,
        policy: str,
    ) -> PolicyFilterBatchResponse:
        decisions_raw = raw.get("decisions")
        if not isinstance(decisions_raw, list):
            raise PolicyFilterError(
                "policy_filter_invalid_json",
                "Chemistry filter response did not include a valid decisions list.",
                self.last_response_preview,
            )
        try:
            decisions = [PolicyFilterDecisionPayload.model_validate(item) for item in decisions_raw]
        except Exception as exc:
            raise PolicyFilterError(
                "policy_filter_invalid_json",
                f"Chemistry filter decision validation failed: {exc}",
                self.last_response_preview,
            ) from exc

        batch = PolicyFilterBatchResponse(
            policy_name=str(raw.get("policy_name") or policy),
            decisions=decisions,
        )
        expected_ids = {candidate.material_id for candidate in payload.candidates}
        seen_ids: set[str] = set()
        for decision in batch.decisions:
            if decision.material_id not in expected_ids:
                raise PolicyFilterError(
                    "policy_filter_unknown_material_id",
                    "Chemistry filter returned a decision for an unknown candidate.",
                    self.last_response_preview,
                )
            if decision.material_id in seen_ids:
                raise PolicyFilterError(
                    "policy_filter_duplicate_decisions",
                    "Chemistry filter returned duplicate candidate decisions.",
                    self.last_response_preview,
                )
            seen_ids.add(decision.material_id)
            if len(decision.reasons) > _MAX_REASONS:
                raise PolicyFilterError(
                    "policy_filter_too_many_reasons",
                    "Chemistry filter returned too many reasons for a candidate.",
                    self.last_response_preview,
                )
            for reason in decision.reasons:
                if len(reason) > _MAX_REASON_CHARS:
                    raise PolicyFilterError(
                        "policy_filter_reason_too_long",
                        "Chemistry filter returned a reason that is too long.",
                        self.last_response_preview,
                    )

        if seen_ids != expected_ids:
            raise PolicyFilterError(
                "policy_filter_missing_candidate_decisions",
                "Chemistry filter did not return decisions for every candidate.",
                self.last_response_preview,
            )
        return batch

    @staticmethod
    def _policy_name(payload: PolicyFilterInput) -> str:
        plan = payload.discovery_plan
        if (
            plan.application_intent == "practical_screening"
            or plan.practicality_mode == "applied"
        ):
            return "practical_screening"
        return "exploratory_screening"

    def _prompt_payload(self, payload: PolicyFilterInput, policy: str) -> dict[str, Any]:
        return {
            "research_goal": payload.discovery_plan.research_goal_raw,
            "plan_summary": {
                "task_class": payload.discovery_plan.task_class,
                "application_intent": payload.discovery_plan.application_intent,
                "material_class": payload.discovery_plan.material_class,
                "practicality_mode": payload.discovery_plan.practicality_mode,
                "ranking_intent": payload.discovery_plan.ranking_intent,
                "policy_name": policy,
            },
            "candidates": [self._candidate_payload(candidate) for candidate in payload.candidates],
        }

    @staticmethod
    def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
        elements = candidate.features.get("elements")
        if not isinstance(elements, list) or not elements:
            elements = sorted(MPRetriever._extract_elements(candidate.formula))
        return {
            "material_id": candidate.material_id,
            "formula": candidate.formula,
            "elements": [str(token) for token in elements],
            "mp_band_gap_ev": candidate.features.get("mp_band_gap_ev"),
            "mp_energy_above_hull": candidate.features.get("mp_energy_above_hull"),
            "nsites": candidate.features.get("nsites"),
            "source": candidate.source,
        }

    @staticmethod
    def _system_prompt(policy: str) -> str:
        return (
            "You are a strict chemistry screening judge for materials discovery.\n"
            "Use only provided metadata. Do not invent missing properties.\n"
            "Task: decide whether each candidate should be kept for downstream band-gap screening.\n"
            "Conservative posture: drop only clear mismatches for requested intent.\n"
            f"Active policy: {policy}.\n"
            "Practical policy: reject clear practical mismatches such as radioactive chemistries, obvious molecular salts,"
            " gas-like or non-bulk-like candidates when user asks for practical bulk semiconductors.\n"
            "Exploratory policy: keep unusual inorganic candidates unless they are clear task mismatches.\n"
            "Return JSON only with shape: "
            '{"policy_name":"...","decisions":[{"material_id":"...","keep":true,"reasons":["..."]}]}\n'
            "Reasons must be short plain strings.\n"
            "Example practical:\n"
            '{"policy_name":"practical_screening","decisions":[{"material_id":"c1","keep":false,"reasons":["radioactive fluoride not practical semiconductor"]},{"material_id":"c2","keep":true,"reasons":[]}]}\n'
            "Example exploratory:\n"
            '{"policy_name":"exploratory_screening","decisions":[{"material_id":"c3","keep":true,"reasons":[]},{"material_id":"c4","keep":false,"reasons":["gas-like species mismatches bulk semiconductor goal"]}]}\n'
        )

    @staticmethod
    def _is_timeout_exc(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        return "timeout" in name or "timed out" in text or "timeout" in text
