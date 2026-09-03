from matsci_agent.nlp.parser import (
    PRIMARY_LLM_MODEL,
    LLMConstraintParser,
    merge_constraints,
    resolve_llm_api_key_env,
    resolve_llm_base_url,
    resolve_llm_model,
)
from matsci_agent.schemas import DiscoveryConstraints


def test_llm_parser_maps_structured_output_to_constraints():
    parser = LLMConstraintParser(
        inference_fn=lambda _goal: {
            "requested_material_class": "nitride",
            "banned_elements": ["Si", "co"],
            "required_elements": ["N"],
            "min_band_gap_ev": 2.0,
            "calculate_matgl": True,
            "top_k": 7,
        }
    )
    parsed = parser.parse("any goal")

    assert parsed.requested_material_class == "nitride"
    assert "Si" in parsed.constraints.banned_elements
    assert "Co" in parsed.constraints.banned_elements
    assert parsed.constraints.required_elements == ["N"]
    assert parsed.constraints.min_band_gap_ev == 2.0
    assert parsed.constraints.calculate_matgl is True
    assert parsed.constraints.top_k == 7
    assert parsed.constraints.max_energy_above_hull is None


def test_omitted_hull_threshold_remains_unset_through_constraint_merge():
    parsed = LLMConstraintParser(inference_fn=lambda _goal: {}).parse("find oxides")
    merged = merge_constraints(DiscoveryConstraints(), parsed.constraints, explicit_base_fields=set())

    assert parsed.constraints.max_energy_above_hull is None
    assert merged.max_energy_above_hull is None


def test_shared_openai_proxy_settings_override_legacy_openrouter_model(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.setenv("MATSCI_OPENROUTER_MODEL", "legacy-model")
    monkeypatch.delenv("MATSCI_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MATSCI_NLP_MODEL", raising=False)
    monkeypatch.delenv("MATSCI_LLM_MODEL", raising=False)
    monkeypatch.delenv("MATSCI_LLM_API_KEY_ENV", raising=False)

    assert resolve_llm_model() == PRIMARY_LLM_MODEL
    assert resolve_llm_base_url() == "https://proxy.example/v1"
    assert resolve_llm_api_key_env() == "OPENAI_API_KEY"


def test_merge_constraints_preserves_explicit_user_values():
    base = DiscoveryConstraints(min_band_gap_ev=3.0, calculate_matgl=False)
    parsed = DiscoveryConstraints(min_band_gap_ev=2.0, calculate_matgl=True, top_k=9)

    merged = merge_constraints(
        base,
        parsed,
        explicit_base_fields={"min_band_gap_ev", "calculate_matgl", "top_k"},
    )

    assert merged.min_band_gap_ev == 3.0
    assert merged.calculate_matgl is False
    assert merged.top_k == 5


def test_llm_parser_accepts_fenced_json_response():
    parser = LLMConstraintParser(
        provider="openrouter",
        inference_fn=lambda _goal: LLMConstraintParser._safe_json_dict(
            "```json\n"
            "{\n"
            '  "requested_material_class": "semiconductor",\n'
            '  "banned_elements": ["Si"],\n'
            '  "required_elements": [],\n'
            '  "min_band_gap_ev": 1.0,\n'
            '  "calculate_matgl": false,\n'
            '  "top_k": 7\n'
            "}\n"
            "```"
        ),
    )
    parsed = parser.parse("any goal")
    assert parsed.requested_material_class == "semiconductor"
    assert parsed.constraints.banned_elements == ["Si"]
    assert parsed.constraints.min_band_gap_ev == 1.0
    assert parsed.constraints.calculate_matgl is False
    assert parsed.constraints.top_k == 7


def test_merge_constraints_applies_parsed_top_k_when_not_explicitly_set():
    base = DiscoveryConstraints()
    parsed = DiscoveryConstraints(top_k=11)

    merged = merge_constraints(base, parsed, explicit_base_fields=set())

    assert merged.top_k == 11


def test_parser_deterministic_control_hints_override_missing_llm_fields():
    parser = LLMConstraintParser(inference_fn=lambda _goal: {})
    parsed = parser.parse(
        "Find semiconductors. Show me top 7 results and do not recalculate with MatGL."
    )
    assert parsed.requested_material_class == "unknown"
    assert parsed.constraints.top_k == 7
    assert parsed.constraints.calculate_matgl is False


def test_parser_deterministic_control_hints_detect_positive_recalc():
    parser = LLMConstraintParser(inference_fn=lambda _goal: {"requested_material_class": "oxide", "top_k": 3})
    parsed = parser.parse("Find oxides and recalculate with MatGL.")
    assert parsed.requested_material_class == "oxide"
    assert parsed.constraints.top_k == 3
    assert parsed.constraints.calculate_matgl is True


def test_parser_normalizes_requested_material_class_to_snake_case():
    parser = LLMConstraintParser(
        inference_fn=lambda _goal: {"requested_material_class": "Amine Solvent"}
    )
    parsed = parser.parse("Find carbon capture amine solvent")

    assert parsed.requested_material_class == "amine_solvent"


def test_parser_keeps_unknown_when_no_strong_class_cue():
    parser = LLMConstraintParser(inference_fn=lambda _goal: {})
    parsed = parser.parse("Find materials with band gap above 2 eV")

    assert parsed.requested_material_class == "unknown"


def test_llm_parser_preserves_nested_mp_filters_for_generic_screening():
    parser = LLMConstraintParser(
        inference_fn=lambda _goal: {
            "mp_filters": {
                "has_props": ["dielectric"],
                "crystal_system": "cubic",
                "formation_energy": {"max": -1.0},
                "density": {"max": 5.0},
                "volume": {"max": 150.0},
            }
        }
    )

    parsed = parser.parse("find cubic low-density dielectric materials")

    assert parsed.constraints.mp_filters.has_props == ["dielectric"]
    assert parsed.constraints.mp_filters.crystal_system == "cubic"
    assert parsed.constraints.mp_filters.formation_energy.max == -1.0
    assert parsed.constraints.mp_filters.density.max == 5.0
    assert parsed.constraints.mp_filters.volume.max == 150.0


def test_legacy_constraint_aliases_merge_into_nested_mp_filters():
    parser = LLMConstraintParser(
        inference_fn=lambda _goal: {
            "required_elements": ["O"],
            "banned_elements": ["Co"],
            "min_band_gap_ev": 2.0,
            "max_energy_above_hull": 0.05,
        }
    )

    parsed = parser.parse("find oxides")

    assert parsed.constraints.mp_filters.elements == ["O"]
    assert parsed.constraints.mp_filters.exclude_elements == ["Co"]
    assert parsed.constraints.mp_filters.band_gap.min == 2.0
    assert parsed.constraints.mp_filters.energy_above_hull.max == 0.05
