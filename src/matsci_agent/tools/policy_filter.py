from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from matsci_agent.nlp.parser import (
    LLMConstraintParser,
    normalize_llm_provider,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_model,
)
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

_POLICY_NAME = "chemistry_screening"
_MAX_REASONS = 3
_MAX_REASON_CHARS = 160
_MAX_RESPONSE_TOKENS = 800
_TIMEOUT_SECS = 20.0
_MAX_OPENROUTER_ATTEMPTS = 3
_IM_PRACTICAL_ELEMENTS = {
    "ac",
    "am",
    "bk",
    "cf",
    "cm",
    "es",
    "fm",
    "md",
    "no",
    "np",
    "pa",
    "pm",
    "po",
    "pu",
    "ra",
    "rn",
    "tc",
    "th",
    "u",
}


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
        self.provider = normalize_llm_provider(provider or os.getenv("MATSCI_NLP_PROVIDER", "openrouter"))
        self.model = resolve_llm_model(model)
        self.inference_fn = inference_fn
        self.last_response_preview: str = ""

    def skip(self, payload: PolicyFilterInput, policy: str) -> PolicyFilterOutput:
        records: list[PolicyFilterRecord] = []
        for candidate in payload.candidates:
            candidate.features["filter_passed"] = True
            candidate.features["filter_reasons"] = []
            candidate.features["filter_policy"] = policy
            candidate.features["filter_source"] = "policy_skipped"
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
            input_payload=self._prompt_payload(payload),
            output_summary={
                "policy": policy,
                "provider": self.provider,
                "model": self.model,
                "decision_backend": "policy_skipped",
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

    def run(self, payload: PolicyFilterInput) -> PolicyFilterOutput:
        if payload.discovery_plan.task_class != "band_gap_screening":
            return self.skip(payload, policy="skipped_non_band_gap_task")

        raw = self._call_llm(payload)
        batch = self._validate_batch_response(raw, payload)
        records: list[PolicyFilterRecord] = []
        filtered: list[Candidate] = []
        by_id = {candidate.material_id: candidate for candidate in payload.candidates}

        for decision in batch.decisions:
            candidate = by_id[decision.material_id]
            keep, reasons = self._apply_hard_reject_guardrail(candidate, decision.keep, list(decision.reasons))
            candidate.features["filter_passed"] = keep
            candidate.features["filter_reasons"] = reasons
            candidate.features["filter_policy"] = _POLICY_NAME
            candidate.features["filter_source"] = "llm"
            record = PolicyFilterRecord(
                candidate=candidate,
                passed=keep,
                reasons=reasons,
                policy=_POLICY_NAME,
            )
            records.append(record)
            if keep:
                filtered.append(candidate)

        provenance = ToolCallProvenance(
            tool_name="policy_filter",
            input_payload=self._prompt_payload(payload),
            output_summary={
                "policy": _POLICY_NAME,
                "provider": self.provider,
                "model": self.model,
                "decision_backend": "llm",
                "batch_size": len(payload.candidates),
                "filtered_count": len(filtered),
                "excluded_count": len(payload.candidates) - len(filtered),
                "raw_response_preview": self.last_response_preview,
            },
        )
        return PolicyFilterOutput(filtered_candidates=filtered, records=records, provenance=provenance)

    def _call_llm(self, payload: PolicyFilterInput) -> dict[str, Any]:
        self.last_response_preview = ""
        if self.inference_fn is not None:
            result = self.inference_fn(self._prompt_payload(payload))
            if isinstance(result, str):
                self.last_response_preview = result[:500]
                return self._parse_response_json(result)
            if isinstance(result, dict):
                self.last_response_preview = json.dumps(result)[:500]
                return result
            raise PolicyFilterError("policy_filter_invalid_json", "Policy filter returned unsupported response type.")

        if self.provider == "openrouter":
            return self._call_openrouter(payload)
        raise PolicyFilterError("policy_filter_llm_request_failed", f"Unsupported provider '{self.provider}'.")

    def _call_openrouter(self, payload: PolicyFilterInput) -> dict[str, Any]:
        api_key, api_key_env = resolve_llm_api_key()
        if not api_key:
            raise PolicyFilterError("policy_filter_llm_request_failed", f"OpenRouter credentials missing: {api_key_env}.")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise PolicyFilterError("policy_filter_llm_request_failed", f"OpenAI client import failed: {exc}") from exc

        client = OpenAI(api_key=api_key, base_url=resolve_llm_base_url(), timeout=_TIMEOUT_SECS)
        last_error: PolicyFilterError | None = None
        for attempt in range(1, _MAX_OPENROUTER_ATTEMPTS + 1):
            use_response_format = attempt == 1
            try:
                response = self._request_openrouter_completion(
                    client=client,
                    payload=payload,
                    use_response_format=use_response_format,
                )
                content = self._extract_completion_content(response)
                self.last_response_preview = content[:500]
                return self._parse_response_json(content)
            except PolicyFilterError as exc:
                last_error = exc
                if exc.code == "policy_filter_timeout":
                    break
            except Exception as exc:
                code = "policy_filter_timeout" if self._is_timeout_exc(exc) else "policy_filter_llm_request_failed"
                last_error = PolicyFilterError(
                    code,
                    f"Policy filter OpenRouter request failed: {exc}",
                    self.last_response_preview,
                )
                if code == "policy_filter_timeout":
                    break
        if last_error is not None:
            raise last_error
        raise PolicyFilterError("policy_filter_llm_request_failed", "Policy filter OpenRouter request failed.")

    def _request_openrouter_completion(
        self,
        *,
        client: Any,
        payload: PolicyFilterInput,
        use_response_format: bool,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": _MAX_RESPONSE_TOKENS,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": json.dumps(self._prompt_payload(payload))},
            ],
        }
        if use_response_format:
            request_kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**request_kwargs)

    @staticmethod
    def _extract_completion_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise PolicyFilterError("policy_filter_invalid_json", "Policy filter response missing choices.")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise PolicyFilterError("policy_filter_invalid_json", "Policy filter response missing message.")
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                return "\n".join(parts)
        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal.strip():
            return refusal
        raise PolicyFilterError("policy_filter_invalid_json", "Policy filter response missing content.")

    @staticmethod
    def _parse_response_json(content: str) -> dict[str, Any]:
        parsed = LLMConstraintParser._safe_json_dict(content)
        if not parsed:
            raise PolicyFilterError("policy_filter_invalid_json", "Policy filter returned invalid JSON.", content[:500])
        return parsed

    def _validate_batch_response(self, raw: dict[str, Any], payload: PolicyFilterInput) -> PolicyFilterBatchResponse:
        decisions_raw = raw.get("decisions")
        if not isinstance(decisions_raw, list):
            raise PolicyFilterError("policy_filter_invalid_json", "Policy filter response missing decisions list.", self.last_response_preview)
        try:
            decisions = [PolicyFilterDecisionPayload.model_validate(item) for item in decisions_raw]
        except Exception as exc:
            raise PolicyFilterError("policy_filter_invalid_json", f"Policy decision validation failed: {exc}", self.last_response_preview) from exc

        expected_ids = {candidate.material_id for candidate in payload.candidates}
        seen_ids: set[str] = set()
        for decision in decisions:
            if decision.material_id not in expected_ids:
                raise PolicyFilterError("policy_filter_unknown_material_id", "Policy filter returned unknown candidate id.", self.last_response_preview)
            if decision.material_id in seen_ids:
                raise PolicyFilterError("policy_filter_duplicate_decisions", "Policy filter returned duplicate candidate decisions.", self.last_response_preview)
            seen_ids.add(decision.material_id)
            decision.reasons = [
                self._normalize_reason(reason)
                for reason in decision.reasons[:_MAX_REASONS]
            ]
        if seen_ids != expected_ids:
            raise PolicyFilterError("policy_filter_missing_candidate_decisions", "Policy filter did not return decisions for every candidate.", self.last_response_preview)
        return PolicyFilterBatchResponse(policy_name=str(raw.get("policy_name") or _POLICY_NAME), decisions=decisions)

    @staticmethod
    def _normalize_reason(reason: str) -> str:
        normalized = " ".join(reason.split())
        if len(normalized) <= _MAX_REASON_CHARS:
            return normalized
        return normalized[: _MAX_REASON_CHARS - 1].rstrip() + "…"

    def _prompt_payload(self, payload: PolicyFilterInput) -> dict[str, Any]:
        return {
            "research_goal": payload.discovery_plan.research_goal_raw,
            "discovery_context": {
                "task_class": payload.discovery_plan.task_class,
                "source_universe": payload.discovery_plan.source_universe,
                "requested_material_class": payload.discovery_plan.requested_material_class,
                "legacy_constraints": {
                    "banned_elements": payload.discovery_plan.parsed_constraints.banned_elements,
                    "required_elements": payload.discovery_plan.parsed_constraints.required_elements,
                    "min_band_gap_ev": payload.discovery_plan.parsed_constraints.min_band_gap_ev,
                    "max_energy_above_hull": payload.discovery_plan.parsed_constraints.max_energy_above_hull,
                },
                "mp_filters": payload.discovery_plan.parsed_constraints.mp_filters.model_dump(mode="json", exclude_defaults=True, exclude_none=True),
            },
            "hard_reject_policy": {
                "impractical_elements": sorted(_IM_PRACTICAL_ELEMENTS),
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
            "is_metal": candidate.features.get("is_metal"),
            "theoretical": candidate.features.get("theoretical"),
            "deprecated": candidate.features.get("deprecated"),
            "crystal_system": candidate.features.get("crystal_system"),
            "spacegroup_symbol": candidate.features.get("spacegroup_symbol"),
            "spacegroup_number": candidate.features.get("spacegroup_number"),
            "has_multiple_entries": candidate.has_multiple_entries,
            "entry_count": candidate.entry_count,
            "source": candidate.source,
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a chemist and materials scientist screening candidate Materials Project entries.\n"
            "Return JSON only.\n"
            "Task: infer user intent from research_goal and discovery_context, then evaluate each candidate against "
            "that intent, chemistry plausibility, and materials knowledge.\n"
            "Use requested_material_class as parser hint only; research_goal is authoritative.\n"
            "Do not assume candidates belong to any preexisting material class beyond source_universe.\n"
            "Use only provided metadata and formula-level chemistry. Do not invent missing properties.\n"
            "Do not keep a candidate only because it has requested elements or numeric property values.\n"
            "Reject candidates whose formula and metadata indicate a different chemical/material concept than the user's request, "
            "including molecular complexes, salts, adducts, or ligand-rich compounds when the user asks for an extended solid material.\n"
            "When the request asks for semiconductors, do not treat every nonmetal or high-band-gap insulator as a semiconductor; "
            "reject salt-like, phosphate/silicate/carbonate-like, hydrate/organic-ligand-rich, or molecular formulas unless the "
            "research_goal explicitly asks for that chemistry.\n"
            "Always reject candidates containing any listed impractical elements.\n"
            "For other candidates, keep only candidates that match the user-requested material concept and constraints.\n"
            "Return exactly this shape: "
            "{\"policy_name\":\"chemistry_screening\",\"decisions\":[{\"material_id\":\"...\",\"keep\":true,\"reasons\":[\"...\"]}]}\n"
            "Reasons must be short plain strings.\n"
            "Include one decision for every candidate exactly once.\n"
        )

    @staticmethod
    def _apply_hard_reject_guardrail(candidate: Candidate, keep: bool, reasons: list[str]) -> tuple[bool, list[str]]:
        elements = candidate.features.get("elements")
        if not isinstance(elements, list) or not elements:
            elements = sorted(MPRetriever._extract_elements(candidate.formula))
        lower_elements = {str(token).lower() for token in elements}
        if lower_elements & _IM_PRACTICAL_ELEMENTS:
            guard_reason = "contains impractical element"
            if guard_reason not in reasons:
                reasons = [guard_reason, *reasons][: _MAX_REASONS]
            return False, reasons
        return keep, reasons[:_MAX_REASONS]

    @staticmethod
    def _is_timeout_exc(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        return "timeout" in name or "timed out" in text or "timeout" in text
