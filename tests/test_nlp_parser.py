from matsci_agent.nlp.parser import LLMConstraintParser, merge_constraints
from matsci_agent.schemas import DiscoveryConstraints


def test_llm_parser_maps_structured_output_to_constraints():
    parser = LLMConstraintParser(
        inference_fn=lambda _goal: {
            "banned_elements": ["Si", "co"],
            "required_elements": ["N"],
            "min_band_gap_ev": 2.0,
            "calculate_matgl": True,
            "top_k": 7,
        }
    )
    parsed = parser.parse("any goal")

    assert "Si" in parsed.banned_elements
    assert "Co" in parsed.banned_elements
    assert parsed.required_elements == ["N"]
    assert parsed.min_band_gap_ev == 2.0
    assert parsed.calculate_matgl is True
    assert parsed.top_k == 7


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
        provider="anthropic",
        inference_fn=lambda _goal: LLMConstraintParser._safe_json_dict(
            "```json\n"
            "{\n"
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
    assert parsed.banned_elements == ["Si"]
    assert parsed.min_band_gap_ev == 1.0
    assert parsed.calculate_matgl is False
    assert parsed.top_k == 7


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
    assert parsed.top_k == 7
    assert parsed.calculate_matgl is False


def test_parser_deterministic_control_hints_detect_positive_recalc():
    parser = LLMConstraintParser(inference_fn=lambda _goal: {"top_k": 3})
    parsed = parser.parse("Find oxides and recalculate with MatGL.")
    assert parsed.top_k == 3
    assert parsed.calculate_matgl is True
