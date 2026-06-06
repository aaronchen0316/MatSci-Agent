from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from matsci_agent.nlp.parser import (
    LLMConstraintParser,
    _ELEMENT_NAME_TO_SYMBOL,
    normalize_llm_provider,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_model,
)
from matsci_agent.schemas import (
    DiscoveryConstraints,
    SearchSpaceExpansionInput,
    SearchSpaceExpansionOutput,
    SearchSpaceTarget,
    ToolCallProvenance,
)

_MAX_RESPONSE_TOKENS = 1800
_TIMEOUT_SECS = 30.0
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_VALID_ELEMENTS = set(_ELEMENT_NAME_TO_SYMBOL.values())
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


class SearchSpaceExpansionError(RuntimeError):
    def __init__(self, code: str, message: str, raw_response_preview: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.raw_response_preview = raw_response_preview[:500]


class SearchSpaceExpansionAgent:
    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        inference_fn: Callable[[dict[str, Any]], dict[str, Any] | str] | None = None,
    ) -> None:
        self.provider = normalize_llm_provider(provider)
        self.model = resolve_llm_model(model)
        self.inference_fn = inference_fn
        self.last_response_preview = ""

    def expand(self, payload: SearchSpaceExpansionInput) -> SearchSpaceExpansionOutput:
        raw = self._call_llm(payload)
        targets = self._validate_targets(raw, payload.discovery_plan.parsed_constraints)
        if not targets:
            raise SearchSpaceExpansionError(
                "search_space_expansion_empty",
                "Search-space expansion returned zero valid MP-compatible formula targets.",
                self.last_response_preview,
            )
        provenance = ToolCallProvenance(
            tool_name="search_space_expander",
            input_payload=self._prompt_payload(payload),
            output_summary={
                "provider": self.provider,
                "model": self.model,
                "target_count": len(targets),
                "raw_response_preview": self.last_response_preview,
            },
        )
        return SearchSpaceExpansionOutput(targets=targets, provenance=provenance)

    def _call_llm(self, payload: SearchSpaceExpansionInput) -> dict[str, Any]:
        self.last_response_preview = ""
        if self.inference_fn is not None:
            result = self.inference_fn(self._prompt_payload(payload))
            if isinstance(result, str):
                self.last_response_preview = result[:500]
                return self._parse_json(result)
            if isinstance(result, dict):
                self.last_response_preview = json.dumps(result)[:500]
                return result
            raise SearchSpaceExpansionError(
                "search_space_expansion_invalid_json",
                "Search-space expansion returned unsupported response type.",
            )
        if self.provider == "openrouter":
            return self._call_openrouter(payload)
        raise SearchSpaceExpansionError(
            "search_space_expansion_request_failed",
            f"Unsupported provider '{self.provider}'.",
        )

    def _call_openrouter(self, payload: SearchSpaceExpansionInput) -> dict[str, Any]:
        api_key, api_key_env = resolve_llm_api_key()
        if not api_key:
            raise SearchSpaceExpansionError(
                "search_space_expansion_request_failed",
                f"OpenRouter credentials missing: {api_key_env}.",
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            raise SearchSpaceExpansionError(
                "search_space_expansion_request_failed",
                f"OpenAI client import failed: {exc}",
            ) from exc

        client = OpenAI(api_key=api_key, base_url=resolve_llm_base_url(), timeout=_TIMEOUT_SECS)
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=_MAX_RESPONSE_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": json.dumps(self._prompt_payload(payload))},
                ],
            )
            content = response.choices[0].message.content or "{}"
            self.last_response_preview = content[:500]
            return self._parse_json(content)
        except SearchSpaceExpansionError:
            raise
        except Exception as exc:
            raise SearchSpaceExpansionError(
                "search_space_expansion_request_failed",
                f"Search-space expansion OpenRouter request failed: {exc}",
                self.last_response_preview,
            ) from exc

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        parsed = LLMConstraintParser._safe_json_dict(content)
        if not parsed:
            raise SearchSpaceExpansionError(
                "search_space_expansion_invalid_json",
                "Search-space expansion returned invalid JSON.",
                content[:500],
            )
        return parsed

    @classmethod
    def _validate_targets(
        cls,
        raw: dict[str, Any],
        constraints: DiscoveryConstraints,
    ) -> list[SearchSpaceTarget]:
        targets_raw = raw.get("formula_targets")
        if not isinstance(targets_raw, list):
            raise SearchSpaceExpansionError(
                "search_space_expansion_invalid_json",
                "Search-space expansion response missing formula_targets list.",
            )

        banned = {element.lower() for element in constraints.banned_elements}
        required = {element.lower() for element in constraints.required_elements}
        seen: set[str] = set()
        targets: list[SearchSpaceTarget] = []
        for item in targets_raw:
            if not isinstance(item, dict):
                continue
            normalized = cls.normalize_formula(str(item.get("formula") or ""))
            if not normalized or normalized in seen:
                continue
            elements = cls.extract_elements(normalized)
            lower_elements = {element.lower() for element in elements}
            if not elements or banned & lower_elements:
                continue
            if required and not required.issubset(lower_elements):
                continue
            chemsys = "-".join(sorted(elements))
            item_chemsys = str(item.get("chemsys") or "").strip()
            if item_chemsys:
                parsed_chemsys = sorted(token for token in item_chemsys.split("-") if token in _VALID_ELEMENTS)
                if parsed_chemsys and parsed_chemsys != sorted(elements):
                    continue
            seen.add(normalized)
            targets.append(
                SearchSpaceTarget(
                    formula=normalized,
                    normalized_formula=normalized,
                    chemsys=chemsys,
                    elements=sorted(elements),
                    confidence=cls._coerce_confidence(item.get("confidence")),
                    rationale=cls._normalize_rationale(item.get("rationale")),
                )
            )
        return targets

    @staticmethod
    def normalize_formula(formula: str) -> str:
        text = formula.translate(_SUBSCRIPT_DIGITS)
        text = re.sub(r"\s+", "", text)
        text = text.replace("−", "-")
        if not text or any(char in text for char in "()[]{}+-=,;"):
            return ""
        pos = 0
        parts: list[str] = []
        for match in _FORMULA_TOKEN_RE.finditer(text):
            if match.start() != pos:
                return ""
            element, count = match.groups()
            if element not in _VALID_ELEMENTS:
                return ""
            if count.startswith("0"):
                return ""
            parts.append(f"{element}{count}")
            pos = match.end()
        if pos != len(text) or not parts:
            return ""
        return "".join(parts)

    @staticmethod
    def extract_elements(formula: str) -> list[str]:
        elements = []
        pos = 0
        for match in _FORMULA_TOKEN_RE.finditer(formula):
            if match.start() != pos:
                return []
            element = match.group(1)
            if element not in _VALID_ELEMENTS:
                return []
            elements.append(element)
            pos = match.end()
        if pos != len(formula):
            return []
        return sorted(set(elements))

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        return 0.0

    @staticmethod
    def _normalize_rationale(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:160]

    @staticmethod
    def _prompt_payload(payload: SearchSpaceExpansionInput) -> dict[str, Any]:
        plan = payload.discovery_plan
        return {
            "research_goal": payload.research_goal,
            "target_count": payload.target_count,
            "task_class": plan.task_class,
            "requested_material_class": plan.requested_material_class,
            "constraints": plan.parsed_constraints.model_dump(mode="json", exclude_none=True),
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You expand materials-search intent into bounded Materials Project formula targets.\n"
            "Return JSON only. No prose. No markdown.\n"
            "Use known/canonical materials candidates from model knowledge only. Do not run or imply live literature search.\n"
            "Do not create broad substitution enumerations or hypothetical combinatorial formulas.\n"
            "Stay tightly constrained to research_goal and constraints.\n"
            "Respect banned_elements, required_elements, numeric MP filters, and requested_material_class.\n"
            "Return up to target_count candidates.\n"
            "Every formula must be ASCII MP-compatible, no unicode subscripts, no parentheses, no charges.\n"
            "chemsys must be sorted element symbols joined by hyphen.\n"
            "Return exactly: "
            "{\"formula_targets\":[{\"formula\":\"CsSnI3\",\"chemsys\":\"Cs-I-Sn\","
            "\"confidence\":0.8,\"rationale\":\"lead-free tin halide perovskite\"}]}\n"
        )
