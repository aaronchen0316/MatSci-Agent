from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Callable

from matsci_agent.schemas import DiscoveryConstraints

logger = logging.getLogger(__name__)


class LLMConstraintParser:
    """LLM-based constraint extractor for natural-language materials goals."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        inference_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.provider = (provider or os.getenv("MATSCI_NLP_PROVIDER", "auto")).lower()
        self.model = model or os.getenv("MATSCI_NLP_MODEL", "")
        self.inference_fn = inference_fn
        self.debug = os.getenv("MATSCI_NLP_DEBUG", "").lower() in {"1", "true", "yes"}
        self.last_errors: list[str] = []
        self.last_response_preview: str = ""

    def parse(self, goal: str) -> DiscoveryConstraints:
        self.last_errors = []
        self.last_response_preview = ""
        if self.inference_fn is not None:
            raw = self.inference_fn(goal)
        else:
            raw = self._call_llm(goal)
        # Deterministic overrides for critical controls so user intent is stable
        # even when the provider misses these fields.
        control_hints = self._extract_control_hints(goal)
        if "top_k" in control_hints:
            raw["top_k"] = control_hints["top_k"]
        if "calculate_matgl" in control_hints:
            raw["calculate_matgl"] = control_hints["calculate_matgl"]
        if self.debug and not raw:
            logger.warning("NLP parser returned empty parse. errors=%s", self.last_errors)
        return self._to_constraints(raw)

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "debug": self.debug,
            "last_errors": list(self.last_errors),
            "last_response_preview": self.last_response_preview,
        }

    def _record_error(self, provider: str, stage: str, exc: Exception | str) -> None:
        err = f"{provider}:{stage}:{type(exc).__name__ if isinstance(exc, Exception) else 'Error'}:{exc}"
        self.last_errors.append(err)
        if self.debug:
            logger.warning("NLP parser error [%s]", err)

    def _call_llm(self, goal: str) -> dict[str, Any]:
        provider = self.provider
        if provider == "auto":
            # Prefer Claude when available, but fail over to OpenAI in case of
            # connectivity/quota/provider-level errors.
            if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
                raw = self._call_anthropic(goal)
                if raw:
                    return raw
            if os.getenv("OPENAI_API_KEY"):
                return self._call_openai(goal)
            return {}

        if provider == "anthropic":
            return self._call_anthropic(goal)
        if provider == "openai":
            return self._call_openai(goal)
        return {}

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a materials-science constraint parser for Materials Project screening.\n"
            "Convert a user goal into ONE JSON object.\n"
            "Do not output markdown, code fences, prose, or explanations.\n"
            "Allowed keys only: banned_elements, required_elements, min_band_gap_ev, calculate_matgl, top_k.\n"
            "\n"
            "Field rules:\n"
            "- banned_elements: array of element symbols to exclude.\n"
            "- required_elements: array of element symbols that must be present.\n"
            "- min_band_gap_ev: number or null.\n"
            "- calculate_matgl: boolean.\n"
            "- top_k: integer only when user requests number of results.\n"
            "\n"
            "Element rules:\n"
            "- Use canonical chemical symbols (Si, Co, Al).\n"
            "- Correct obvious element typos from natural language (for example 'Sillicon' -> Si).\n"
            "\n"
            "Critical interpretation rules:\n"
            "- If user says top/show/return N results, set top_k=N.\n"
            "- If user says do not recalculate / no recalc / no MatGL / do not run MatGL, set calculate_matgl=false.\n"
            "- If user explicitly asks to recalculate/recompute/run MatGL, set calculate_matgl=true.\n"
            "- If recalc intent is not mentioned, set calculate_matgl=false.\n"
            "- If result-count intent is not mentioned, omit top_k.\n"
            "\n"
            "Examples:\n"
            "User: Find semiconductors with no silicon, band gap above 1 eV, top 7, do not recalculate.\n"
            "Output: {\"banned_elements\":[\"Si\"],\"required_elements\":[],\"min_band_gap_ev\":1.0,\"calculate_matgl\":false,\"top_k\":7}\n"
            "User: Find oxides with band gap above 2 eV.\n"
            "Output: {\"banned_elements\":[],\"required_elements\":[\"O\"],\"min_band_gap_ev\":2.0,\"calculate_matgl\":false}\n"
        )

    def _call_anthropic(self, goal: str) -> dict[str, Any]:
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self._record_error("anthropic", "missing_api_key", "CLAUDE_API_KEY not set")
            return {}

        try:
            from anthropic import Anthropic
        except Exception as exc:
            self._record_error("anthropic", "import", exc)
            return {}

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if base_url:
            client = Anthropic(api_key=api_key, base_url=base_url)
        else:
            client = Anthropic(api_key=api_key)
        model = self.model or os.getenv(
            "MATSCI_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001-thinking"
        )
        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": goal}],
            )
            parts = []
            for block in getattr(response, "content", []):
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            content = "\n".join(parts) if parts else "{}"
            self.last_response_preview = content[:500]
        except Exception as exc:
            self._record_error("anthropic", "request", exc)
            return {}
        parsed = self._safe_json_dict(content)
        if not parsed:
            self._record_error(
                "anthropic", "parse", "response_did_not_contain_valid_json_object"
            )
        return parsed

    def _call_openai(self, goal: str) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._record_error("openai", "missing_api_key", "OPENAI_API_KEY not set")
            return {}

        try:
            from openai import OpenAI
        except Exception as exc:
            self._record_error("openai", "import", exc)
            return {}

        client = OpenAI(api_key=api_key)
        model = self.model or os.getenv("MATSCI_OPENAI_MODEL", "gpt-4.1-mini")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": goal},
                ],
            )
            content = response.choices[0].message.content or "{}"
            self.last_response_preview = content[:500]
        except Exception as exc:
            self._record_error("openai", "request", exc)
            return {}
        parsed = self._safe_json_dict(content)
        if not parsed:
            self._record_error(
                "openai", "parse", "response_did_not_contain_valid_json_object"
            )
        return parsed

    @staticmethod
    def _safe_json_dict(content: str) -> dict[str, Any]:
        text = content.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict) and parsed:
            return parsed

        # Handle fenced markdown payloads (```json ... ```).
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                inner = "\n".join(lines[1:-1]).strip()
                try:
                    parsed = json.loads(inner)
                except Exception:
                    parsed = {}
                if isinstance(parsed, dict):
                    return parsed

        # Handle surrounding prose with an embedded JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                parsed = {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    @staticmethod
    def _normalize_elements(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for v in values:
            if not isinstance(v, str):
                continue
            token = v.strip()
            if not token:
                continue
            if len(token) <= 2 and token.isalpha():
                out.append(token.capitalize())
            else:
                # Expect the LLM to return symbols; keep non-symbol tokens out.
                continue
        return sorted(set(out))

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _coerce_top_k(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        parsed: int | None = None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        elif isinstance(value, str):
            token = value.strip()
            if token.isdigit():
                parsed = int(token)
        if parsed is None:
            return None
        return max(1, min(100, parsed))

    @staticmethod
    def _extract_control_hints(goal: str) -> dict[str, Any]:
        text = goal.lower()
        hints: dict[str, Any] = {}

        # Parse result count intent from common phrasings.
        top_k_patterns = [
            r"\btop\s+(\d{1,3})\b",
            r"\bshow\s+(?:me\s+)?(\d{1,3})\s+results?\b",
            r"\breturn\s+(\d{1,3})\s+results?\b",
            r"\bonly\s+(\d{1,3})\b",
        ]
        for pattern in top_k_patterns:
            match = re.search(pattern, text)
            if match:
                hints["top_k"] = max(1, min(100, int(match.group(1))))
                break

        negative_recalc_patterns = [
            r"\bdo\s+not\s+recalculat",
            r"\bdon't\s+recalculat",
            r"\bno\s+recalculat",
            r"\bdo\s+not\s+recompute",
            r"\bdon't\s+recompute",
            r"\bno\s+matgl\b",
            r"\bdo\s+not\s+run\s+matgl",
            r"\bdon't\s+run\s+matgl",
        ]
        positive_recalc_patterns = [
            r"\brecalculat",
            r"\brecompute",
            r"\bforce\s+matgl",
            r"\brun\s+matgl",
            r"\bcalculate\s+matgl",
        ]

        has_negative = any(re.search(p, text) for p in negative_recalc_patterns)
        has_positive = any(re.search(p, text) for p in positive_recalc_patterns)

        if has_negative:
            hints["calculate_matgl"] = False
        elif has_positive:
            hints["calculate_matgl"] = True

        return hints

    def _to_constraints(self, raw: dict[str, Any]) -> DiscoveryConstraints:
        constraints = DiscoveryConstraints()
        constraints.banned_elements = self._normalize_elements(raw.get("banned_elements"))
        constraints.required_elements = self._normalize_elements(raw.get("required_elements"))

        min_gap = raw.get("min_band_gap_ev")
        if isinstance(min_gap, (int, float)):
            constraints.min_band_gap_ev = float(min_gap)

        constraints.calculate_matgl = self._coerce_bool(
            raw.get("calculate_matgl"), default=False
        )
        parsed_top_k = self._coerce_top_k(raw.get("top_k"))
        if parsed_top_k is not None:
            constraints.top_k = parsed_top_k
        return constraints


@lru_cache(maxsize=1)
def _default_parser() -> LLMConstraintParser:
    return LLMConstraintParser()


def parse_goal_to_constraints(goal: str) -> DiscoveryConstraints:
    return _default_parser().parse(goal)


def get_parser_debug_snapshot() -> dict[str, Any]:
    return _default_parser().get_debug_snapshot()


def merge_constraints(
    base: DiscoveryConstraints,
    parsed: DiscoveryConstraints,
    explicit_base_fields: set[str] | None = None,
) -> DiscoveryConstraints:
    merged = base.model_copy(deep=True)
    explicit = explicit_base_fields or set()
    defaults = DiscoveryConstraints()

    merged.banned_elements = sorted(
        set(merged.banned_elements) | set(parsed.banned_elements)
    )
    merged.required_elements = sorted(
        set(merged.required_elements) | set(parsed.required_elements)
    )

    if merged.min_band_gap_ev is None and parsed.min_band_gap_ev is not None:
        merged.min_band_gap_ev = parsed.min_band_gap_ev

    if "calculate_matgl" not in explicit:
        merged.calculate_matgl = merged.calculate_matgl or parsed.calculate_matgl
    if "top_k" not in explicit and parsed.top_k != defaults.top_k:
        merged.top_k = parsed.top_k
    return merged
