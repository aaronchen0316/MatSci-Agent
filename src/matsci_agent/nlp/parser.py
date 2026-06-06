from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Callable

from matsci_agent.schemas import DiscoveryConstraints, FloatRange, IntRange, MPFilters, ParsedDiscoveryIntent

logger = logging.getLogger(__name__)
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
_ELEMENT_NAME_TO_SYMBOL = {
    "hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C",
    "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne", "sodium": "Na", "magnesium": "Mg",
    "aluminum": "Al", "aluminium": "Al", "silicon": "Si", "phosphorus": "P", "sulfur": "S", "sulphur": "S",
    "chlorine": "Cl", "argon": "Ar", "potassium": "K", "calcium": "Ca", "scandium": "Sc", "titanium": "Ti",
    "vanadium": "V", "chromium": "Cr", "manganese": "Mn", "iron": "Fe", "cobalt": "Co", "nickel": "Ni",
    "copper": "Cu", "zinc": "Zn", "gallium": "Ga", "germanium": "Ge", "arsenic": "As", "selenium": "Se",
    "bromine": "Br", "krypton": "Kr", "rubidium": "Rb", "strontium": "Sr", "yttrium": "Y", "zirconium": "Zr",
    "niobium": "Nb", "molybdenum": "Mo", "technetium": "Tc", "ruthenium": "Ru", "rhodium": "Rh", "palladium": "Pd",
    "silver": "Ag", "cadmium": "Cd", "indium": "In", "tin": "Sn", "antimony": "Sb", "tellurium": "Te",
    "iodine": "I", "xenon": "Xe", "cesium": "Cs", "caesium": "Cs", "barium": "Ba", "lanthanum": "La",
    "cerium": "Ce", "praseodymium": "Pr", "neodymium": "Nd", "promethium": "Pm", "samarium": "Sm", "europium": "Eu",
    "gadolinium": "Gd", "terbium": "Tb", "dysprosium": "Dy", "holmium": "Ho", "erbium": "Er", "thulium": "Tm",
    "ytterbium": "Yb", "lutetium": "Lu", "hafnium": "Hf", "tantalum": "Ta", "tungsten": "W", "rhenium": "Re",
    "osmium": "Os", "iridium": "Ir", "platinum": "Pt", "gold": "Au", "mercury": "Hg", "thallium": "Tl",
    "lead": "Pb", "bismuth": "Bi", "polonium": "Po", "astatine": "At", "radon": "Rn", "francium": "Fr",
    "radium": "Ra", "actinium": "Ac", "thorium": "Th", "protactinium": "Pa", "uranium": "U", "neptunium": "Np",
    "plutonium": "Pu", "americium": "Am", "curium": "Cm", "berkelium": "Bk", "californium": "Cf", "einsteinium": "Es",
    "fermium": "Fm", "mendelevium": "Md", "nobelium": "No", "lawrencium": "Lr", "rutherfordium": "Rf",
    "dubnium": "Db", "seaborgium": "Sg", "bohrium": "Bh", "hassium": "Hs", "meitnerium": "Mt", "darmstadtium": "Ds",
    "roentgenium": "Rg", "copernicium": "Cn", "nihonium": "Nh", "flerovium": "Fl", "moscovium": "Mc",
    "livermorium": "Lv", "tennessine": "Ts", "oganesson": "Og",
}
_ELEMENT_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(name) for name in _ELEMENT_NAME_TO_SYMBOL), key=len, reverse=True)) + r")\b"
)


def normalize_llm_provider(provider: str | None) -> str:
    normalized = (provider or "openrouter").strip().lower()
    if normalized in {"", "auto", "anthropic", "openai", "openrouter"}:
        return "openrouter"
    return normalized


def resolve_llm_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("MATSCI_NLP_MODEL")
        or os.getenv("MATSCI_LLM_MODEL")
        or os.getenv("MATSCI_OPENROUTER_MODEL")
        or _DEFAULT_OPENROUTER_MODEL
    )


def resolve_llm_base_url() -> str:
    return os.getenv("MATSCI_LLM_BASE_URL", _DEFAULT_OPENROUTER_BASE_URL)


def resolve_llm_api_key_env() -> str:
    configured = os.getenv("MATSCI_LLM_API_KEY_ENV", "").strip()
    if configured:
        return configured
    if os.getenv("OPENROUTER_API_KEY_RAG"):
        return "OPENROUTER_API_KEY_RAG"
    return "OPENROUTER_API_KEY"


def resolve_llm_api_key() -> tuple[str | None, str]:
    env_name = resolve_llm_api_key_env()
    api_key = os.getenv(env_name)
    if api_key:
        return api_key, env_name
    if env_name != "OPENROUTER_API_KEY" and os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY"
    return None, env_name


class LLMConstraintParser:
    """LLM-based constraint extractor for natural-language materials goals."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        inference_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.provider = normalize_llm_provider(provider or os.getenv("MATSCI_NLP_PROVIDER", "openrouter"))
        self.model = resolve_llm_model(model)
        self.inference_fn = inference_fn
        self.debug = os.getenv("MATSCI_NLP_DEBUG", "").lower() in {"1", "true", "yes"}
        self.last_errors: list[str] = []
        self.last_response_preview: str = ""

    def parse(self, goal: str) -> ParsedDiscoveryIntent:
        self.last_errors = []
        self.last_response_preview = ""
        raw = self.inference_fn(goal) if self.inference_fn is not None else self._call_llm(goal)
        control_hints = self._extract_control_hints(goal)
        if "top_k" in control_hints:
            raw["top_k"] = control_hints["top_k"]
        if "calculate_matgl" in control_hints:
            raw["calculate_matgl"] = control_hints["calculate_matgl"]
        if self.debug and not raw:
            logger.warning("NLP parser returned empty parse. errors=%s", self.last_errors)
        return self._to_parsed_intent(raw, goal)

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
        if self.provider == "openrouter":
            return self._call_openrouter(goal)
        self._record_error(self.provider, "provider", "unsupported_provider")
        return {}

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a chemist and materials scientist translating a user request into "
            "Materials Project search intent.\n"
            "Return exactly one JSON object. No markdown. No prose. No code fences.\n"
            "Top-level keys allowed only: requested_material_class, banned_elements, required_elements, min_band_gap_ev, "
            "calculate_matgl, max_energy_above_hull, top_k, mp_filters.\n"
            "Leave omitted/undefined filters out.\n"
            "requested_material_class must be one free-form snake_case label inferred from explicit or strongly implied "
            "user intent. Otherwise use unknown.\n"
            "Use canonical element symbols.\n"
            "Use mp_filters only for real Materials Project summary.search filters.\n"
            "mp_filters keys allowed only: formula, chemsys, material_ids, elements, exclude_elements, "
            "possible_species, has_props, is_metal, is_stable, is_gap_direct, theoretical, deprecated, "
            "has_reconstructed, crystal_system, spacegroup_number, spacegroup_symbol, band_gap, "
            "energy_above_hull, formation_energy, density, efermi, total_magnetization, volume, "
            "num_sites, num_elements.\n"
            "For numeric filters use {\"min\": number, \"max\": number} and omit sides not requested.\n"
            "For formula/chemsys/material_ids/spacegroup_* you may use a string or a list.\n"
            "For list-valued fields return arrays.\n"
            "Map chemistry words into real MP filters when appropriate, but do not invent unsupported fields.\n"
            "Examples:\n"
            "User: Find oxides with band gap above 2 eV and no cobalt.\n"
            "Output: "
            "{\"requested_material_class\":\"oxide\",\"banned_elements\":[\"Co\"],\"required_elements\":[\"O\"],\"min_band_gap_ev\":2.0,"
            "\"calculate_matgl\":false,\"mp_filters\":{\"elements\":[\"O\"],\"exclude_elements\":[\"Co\"],"
            "\"band_gap\":{\"min\":2.0}}}\n"
            "User: Find cubic materials in space group 221 with formula ABO3.\n"
            "Output: "
            "{\"requested_material_class\":\"unknown\",\"calculate_matgl\":false,\"mp_filters\":{\"formula\":\"ABO3\",\"crystal_system\":\"cubic\","
            "\"spacegroup_number\":221}}\n"
            "User: Find carbon capture amine solvent.\n"
            "Output: "
            "{\"requested_material_class\":\"amine_solvent\",\"calculate_matgl\":false}\n"
        )

    def _call_openrouter(self, goal: str) -> dict[str, Any]:
        api_key, api_key_env = resolve_llm_api_key()
        if not api_key:
            self._record_error("openrouter", "missing_api_key", f"{api_key_env} not set")
            return {}

        try:
            from openai import OpenAI
        except Exception as exc:
            self._record_error("openrouter", "import", exc)
            return {}

        client = OpenAI(api_key=api_key, base_url=resolve_llm_base_url())
        try:
            response = client.chat.completions.create(
                model=self.model,
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
            self._record_error("openrouter", "request", exc)
            return {}
        parsed = self._safe_json_dict(content)
        if not parsed:
            self._record_error("openrouter", "parse", "response_did_not_contain_valid_json_object")
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

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _normalize_elements(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            token = value.strip()
            if len(token) <= 2 and token.isalpha():
                out.append(token.capitalize())
        return sorted(set(out))

    @staticmethod
    def _normalize_str_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out = []
        for value in values:
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
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
        if isinstance(value, int):
            return max(1, min(100, value))
        if isinstance(value, float):
            return max(1, min(100, int(value)))
        if isinstance(value, str) and value.strip().isdigit():
            return max(1, min(100, int(value.strip())))
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            token = value.strip()
            if token.lstrip("-").isdigit():
                return int(token)
        return None

    def _coerce_float_range(self, value: Any) -> FloatRange | None:
        if not isinstance(value, dict):
            return None
        lower = self._coerce_float(value.get("min"))
        upper = self._coerce_float(value.get("max"))
        if lower is None and upper is None:
            return None
        return FloatRange(min=lower, max=upper)

    def _coerce_int_range(self, value: Any) -> IntRange | None:
        if not isinstance(value, dict):
            return None
        lower = self._coerce_int(value.get("min"))
        upper = self._coerce_int(value.get("max"))
        if lower is None and upper is None:
            return None
        return IntRange(min=lower, max=upper)

    @staticmethod
    def _coerce_scalar_or_list(value: Any, value_type: type[int] | type[str]) -> str | list[str] | int | list[int] | None:
        if isinstance(value, value_type):
            token = value.strip() if value_type is str else value
            return token if token else None
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, value_type):
                    token = item.strip() if value_type is str else item
                    if token or value_type is int:
                        cleaned.append(token)
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _extract_control_hints(goal: str) -> dict[str, Any]:
        text = goal.lower()
        hints: dict[str, Any] = {}
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

        has_negative = any(re.search(pattern, text) for pattern in negative_recalc_patterns)
        has_positive = any(re.search(pattern, text) for pattern in positive_recalc_patterns)

        if has_negative:
            hints["calculate_matgl"] = False
        elif has_positive:
            hints["calculate_matgl"] = True

        return hints

    @staticmethod
    def _extract_clause_elements(text: str) -> list[str]:
        matches = _ELEMENT_NAME_PATTERN.findall(text.lower())
        return sorted({_ELEMENT_NAME_TO_SYMBOL[name] for name in matches})

    @classmethod
    def _extract_element_constraints(cls, goal: str) -> tuple[list[str], list[str]]:
        text = goal.lower()
        stop = r"(?=\bband gap\b|\benergy above hull\b|\brecalculate\b|\brank\b|\bsort\b|\btop\s+\d+\b|$)"
        required: set[str] = set()
        banned: set[str] = set()
        positive_patterns = [
            rf"\bwith\b\s+(.*?){stop}",
            rf"\bcontaining\b\s+(.*?){stop}",
            rf"\bcontains\b\s+(.*?){stop}",
            rf"\bincluding\b\s+(.*?){stop}",
        ]
        negative_patterns = [
            rf"\bwithout\b\s+(.*?){stop}",
            rf"\bexcluding\b\s+(.*?){stop}",
            rf"\bexclude\b\s+(.*?){stop}",
            rf"\bfree of\b\s+(.*?){stop}",
            rf"\bno\b\s+(.*?){stop}",
        ]
        for pattern in positive_patterns:
            for clause in re.findall(pattern, text):
                required.update(cls._extract_clause_elements(clause))
        for pattern in negative_patterns:
            for clause in re.findall(pattern, text):
                banned.update(cls._extract_clause_elements(clause))
        return sorted(required - banned), sorted(banned)

    @staticmethod
    def _extract_band_gap_bounds(goal: str) -> tuple[float | None, float | None]:
        text = goal.lower()
        min_match = re.search(
            r"\bband gap\b.*?\b(?:above|greater than|higher than|over|at least|>=)\s*(\d+(?:\.\d+)?)",
            text,
        )
        max_match = re.search(
            r"\bband gap\b.*?\b(?:below|less than|lower than|under|at most|<=)\s*(\d+(?:\.\d+)?)",
            text,
        )
        min_gap = float(min_match.group(1)) if min_match else None
        max_gap = float(max_match.group(1)) if max_match else None
        return min_gap, max_gap

    @staticmethod
    def _extract_numeric_bounds(goal: str, label_pattern: str) -> tuple[float | None, float | None]:
        text = goal.lower()
        min_match = re.search(
            rf"\b(?:{label_pattern})\b.*?\b(?:above|greater than|higher than|over|at least|>=)\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        max_match = re.search(
            rf"\b(?:{label_pattern})\b.*?\b(?:below|less than|lower than|under|at most|<=)\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        return (
            float(min_match.group(1)) if min_match else None,
            float(max_match.group(1)) if max_match else None,
        )

    def _coerce_mp_filters(self, raw: dict[str, Any]) -> MPFilters:
        filters_raw = raw.get("mp_filters")
        if not isinstance(filters_raw, dict):
            return MPFilters()
        return MPFilters(
            formula=self._coerce_scalar_or_list(filters_raw.get("formula"), str),
            chemsys=self._coerce_scalar_or_list(filters_raw.get("chemsys"), str),
            material_ids=self._coerce_scalar_or_list(filters_raw.get("material_ids"), str),
            elements=self._normalize_elements(filters_raw.get("elements")),
            exclude_elements=self._normalize_elements(filters_raw.get("exclude_elements")),
            possible_species=self._normalize_str_list(filters_raw.get("possible_species")),
            has_props=self._normalize_str_list(filters_raw.get("has_props")),
            is_metal=filters_raw.get("is_metal") if isinstance(filters_raw.get("is_metal"), bool) else None,
            is_stable=filters_raw.get("is_stable") if isinstance(filters_raw.get("is_stable"), bool) else None,
            is_gap_direct=filters_raw.get("is_gap_direct") if isinstance(filters_raw.get("is_gap_direct"), bool) else None,
            theoretical=filters_raw.get("theoretical") if isinstance(filters_raw.get("theoretical"), bool) else None,
            deprecated=filters_raw.get("deprecated") if isinstance(filters_raw.get("deprecated"), bool) else None,
            has_reconstructed=(
                filters_raw.get("has_reconstructed")
                if isinstance(filters_raw.get("has_reconstructed"), bool)
                else None
            ),
            crystal_system=(
                filters_raw.get("crystal_system").strip().lower()
                if isinstance(filters_raw.get("crystal_system"), str) and filters_raw.get("crystal_system").strip()
                else None
            ),
            spacegroup_number=self._coerce_scalar_or_list(filters_raw.get("spacegroup_number"), int),
            spacegroup_symbol=self._coerce_scalar_or_list(filters_raw.get("spacegroup_symbol"), str),
            band_gap=self._coerce_float_range(filters_raw.get("band_gap")),
            energy_above_hull=self._coerce_float_range(filters_raw.get("energy_above_hull")),
            formation_energy=self._coerce_float_range(filters_raw.get("formation_energy")),
            density=self._coerce_float_range(filters_raw.get("density")),
            efermi=self._coerce_float_range(filters_raw.get("efermi")),
            total_magnetization=self._coerce_float_range(filters_raw.get("total_magnetization")),
            volume=self._coerce_float_range(filters_raw.get("volume")),
            num_sites=self._coerce_int_range(filters_raw.get("num_sites")),
            num_elements=self._coerce_int_range(filters_raw.get("num_elements")),
        )

    @staticmethod
    def _normalize_requested_material_class(value: Any) -> str:
        if isinstance(value, str):
            normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
            if normalized:
                return normalized
        return "unknown"

    def _to_parsed_intent(self, raw: dict[str, Any], goal: str) -> ParsedDiscoveryIntent:
        defaults = DiscoveryConstraints()
        mp_filters = self._coerce_mp_filters(raw)
        extracted_required, extracted_banned = self._extract_element_constraints(goal)
        banned_elements = sorted(
            set(self._normalize_elements(raw.get("banned_elements")) or list(mp_filters.exclude_elements))
            | set(extracted_banned)
        )
        required_elements = sorted(
            (set(self._normalize_elements(raw.get("required_elements")) or list(mp_filters.elements)) | set(extracted_required))
            - set(banned_elements)
        )
        min_gap = self._coerce_float(raw.get("min_band_gap_ev"))
        if min_gap is None and mp_filters.band_gap is not None:
            min_gap = mp_filters.band_gap.min
        extracted_min_gap, extracted_max_gap = self._extract_band_gap_bounds(goal)
        if min_gap is None:
            min_gap = extracted_min_gap
        max_hull = self._coerce_float(raw.get("max_energy_above_hull"))
        if max_hull is None and mp_filters.energy_above_hull is not None and mp_filters.energy_above_hull.max is not None:
            max_hull = mp_filters.energy_above_hull.max
        if mp_filters.elements == [] and required_elements:
            mp_filters.elements = list(required_elements)
        if mp_filters.exclude_elements == [] and banned_elements:
            mp_filters.exclude_elements = list(banned_elements)
        if mp_filters.band_gap is None and (extracted_min_gap is not None or extracted_max_gap is not None):
            mp_filters.band_gap = FloatRange(min=extracted_min_gap, max=extracted_max_gap)
        elif mp_filters.band_gap is not None:
            if mp_filters.band_gap.min is None and extracted_min_gap is not None:
                mp_filters.band_gap.min = extracted_min_gap
            if mp_filters.band_gap.max is None and extracted_max_gap is not None:
                mp_filters.band_gap.max = extracted_max_gap
        for field_name, label_pattern in {
            "formation_energy": r"formation energy",
            "density": r"density",
            "volume": r"volume",
            "energy_above_hull": r"energy above hull|hull energy",
        }.items():
            lower, upper = self._extract_numeric_bounds(goal, label_pattern)
            current = getattr(mp_filters, field_name)
            if current is None and (lower is not None or upper is not None):
                setattr(mp_filters, field_name, FloatRange(min=lower, max=upper))
            elif current is not None:
                if current.min is None and lower is not None:
                    current.min = lower
                if current.max is None and upper is not None:
                    current.max = upper
        parsed_top_k = self._coerce_top_k(raw.get("top_k"))
        requested_material_class = self._normalize_requested_material_class(raw.get("requested_material_class"))
        return ParsedDiscoveryIntent(
            requested_material_class=requested_material_class,
            constraints=DiscoveryConstraints(
                banned_elements=banned_elements,
                required_elements=required_elements,
                min_band_gap_ev=min_gap,
                calculate_matgl=self._coerce_bool(raw.get("calculate_matgl"), default=False),
                max_energy_above_hull=max_hull if max_hull is not None else defaults.max_energy_above_hull,
                top_k=parsed_top_k if parsed_top_k is not None else defaults.top_k,
                mp_filters=mp_filters,
            ),
        )


@lru_cache(maxsize=1)
def _default_parser() -> LLMConstraintParser:
    return LLMConstraintParser()


def parse_goal_to_constraints(goal: str) -> DiscoveryConstraints:
    return _default_parser().parse(goal).constraints


def parse_goal_to_intent(goal: str) -> ParsedDiscoveryIntent:
    return _default_parser().parse(goal)


def get_parser_debug_snapshot() -> dict[str, Any]:
    return _default_parser().get_debug_snapshot()


def _merge_range(base: FloatRange | IntRange | None, parsed: FloatRange | IntRange | None) -> FloatRange | IntRange | None:
    if base is None:
        return parsed
    if parsed is None:
        return base
    merged = base.model_copy(deep=True)
    if getattr(merged, "min") is None:
        merged.min = parsed.min
    if getattr(merged, "max") is None:
        merged.max = parsed.max
    return merged


def merge_constraints(
    base: DiscoveryConstraints,
    parsed: DiscoveryConstraints,
    explicit_base_fields: set[str] | None = None,
) -> DiscoveryConstraints:
    merged = base.model_copy(deep=True)
    explicit = explicit_base_fields or set()
    defaults = DiscoveryConstraints()

    merged.banned_elements = sorted(set(merged.banned_elements) | set(parsed.banned_elements))
    merged.required_elements = sorted(set(merged.required_elements) | set(parsed.required_elements))

    if merged.min_band_gap_ev is None and parsed.min_band_gap_ev is not None:
        merged.min_band_gap_ev = parsed.min_band_gap_ev
    if "calculate_matgl" not in explicit:
        merged.calculate_matgl = merged.calculate_matgl or parsed.calculate_matgl
    if "top_k" not in explicit and parsed.top_k != defaults.top_k:
        merged.top_k = parsed.top_k
    if "max_energy_above_hull" not in explicit and merged.max_energy_above_hull == defaults.max_energy_above_hull:
        if parsed.max_energy_above_hull != defaults.max_energy_above_hull:
            merged.max_energy_above_hull = parsed.max_energy_above_hull

    merged_filters = merged.mp_filters.model_copy(deep=True)
    parsed_filters = parsed.mp_filters
    merged_filters.elements = sorted(set(merged_filters.elements) | set(parsed_filters.elements))
    merged_filters.exclude_elements = sorted(set(merged_filters.exclude_elements) | set(parsed_filters.exclude_elements))
    merged_filters.possible_species = sorted(set(merged_filters.possible_species) | set(parsed_filters.possible_species))
    merged_filters.has_props = sorted(set(merged_filters.has_props) | set(parsed_filters.has_props))

    for field_name in [
        "formula",
        "chemsys",
        "material_ids",
        "is_metal",
        "is_stable",
        "is_gap_direct",
        "theoretical",
        "deprecated",
        "has_reconstructed",
        "crystal_system",
        "spacegroup_number",
        "spacegroup_symbol",
    ]:
        if getattr(merged_filters, field_name) is None:
            setattr(merged_filters, field_name, getattr(parsed_filters, field_name))

    for field_name in [
        "band_gap",
        "energy_above_hull",
        "formation_energy",
        "density",
        "efermi",
        "total_magnetization",
        "volume",
        "num_sites",
        "num_elements",
    ]:
        setattr(
            merged_filters,
            field_name,
            _merge_range(getattr(merged_filters, field_name), getattr(parsed_filters, field_name)),
        )

    merged.mp_filters = merged_filters
    return merged
