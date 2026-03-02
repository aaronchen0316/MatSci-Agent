# MatSci-Agent Notes (Common Issues and Interventions)

## Scope
This file records practical debugging issues encountered during development, plus the exact interventions that resolved them.

## 1) Environment Variable Visibility Mismatch
### Symptom
- API keys exported in VSCode terminal were not visible inside Codex runtime commands.

### Root Cause
- Different process/session environments.

### Intervention
- Verify with shell checks (`echo ${VAR:+set}`) in the same runtime that executes commands.
- Persist env vars in `~/.zshrc` when needed.

## 2) OpenAI Quota Error During NLP Parsing
### Symptom
- `openai.RateLimitError` with `insufficient_quota`.

### Root Cause
- ChatGPT subscription does not provide OpenAI API credits.

### Intervention
- Added provider abstraction in parser and Anthropic support.
- Avoided hard failure by returning neutral parse on provider errors with debug logging.

## 3) Anthropic Third-Party Gateway Model Mismatch
### Symptom
- `model_not_found` from gateway for `claude-3-5-sonnet-latest`.

### Root Cause
- Provider channel did not expose that alias.

### Intervention
- Set Anthropic default model to `claude-3-5-sonnet-20241022`.
- Restored parser support for `ANTHROPIC_BASE_URL`.

## 4) Parser Returned Empty Parse Despite Claude Usage
### Symptom
- Debug output showed:
  - `provider='anthropic'`
  - `last_errors=[]`
  - `last_response_preview` contained fenced JSON
  - parsed constraints were empty.

### Root Cause
- Parser bug in `_safe_json_dict`: early return occurred before fenced-JSON extraction path.

### Intervention
- Fixed `_safe_json_dict` to continue parsing when direct JSON parse yields empty dict.
- Added support for:
  - plain JSON
  - fenced JSON (```json ... ```)
  - embedded JSON in surrounding text
- Added regression test for fenced JSON parsing.

## 5) Retrieval Returned Chemically Implausible Result (`O2` as semiconductor)
### Symptom
- Query for semiconductors returned elemental `O2` candidate.

### Root Cause
- Missing domain-semantic filter for semiconductor intent.
- NLP typo handling for "Sillicon" needed improvement.

### Intervention
- Added parser typo/name handling to normalize silicon typo to `Si`.
- Added retriever goal-semantic filter:
  - if goal includes "semiconductor", exclude single-element candidates.
- Added tests for both typo parsing and semiconductor filtering.

## 6) Empty Candidate Lists on Live MP Retrieval
### Symptom
- `/discover` returned empty candidates when live MP path returned zero entries.

### Root Cause
- Retriever treated empty live response as success and skipped fallback.

### Intervention
- Changed retriever policy:
  - only accept live result when candidate list is non-empty.
  - otherwise fallback to mock with explicit provenance source.

## 7) Output Too Verbose for End Users
### Symptom
- API returned full internal candidate payload/provenance by default.

### Intervention
- Added summary response model for `/discover` that returns only:
  - `material_id`
  - `formula`
  - `band_gap_ev`

## 8) Practical Debug Procedure (Reusable)
1. Turn on parser debug: `MATSCI_NLP_DEBUG=1`.
2. Run parser directly and inspect:
   - parsed constraints
   - `get_parser_debug_snapshot()`
3. Confirm provider/model/env in same shell process as `uv run`.
4. Verify gateway model list locally if DNS/network is restricted in tooling runtime.
5. Add a regression test for every production parsing edge case discovered.

## 9) Security Note
- API keys must not be shared in chat transcripts.
- If exposed, rotate immediately and update local env safely.

## 10) Predictor Determinism Issue (Recalc Cap)
### Symptom
- With `calculate_matgl=true` and recalc cap active, selected candidates could vary with input ordering.

### Root Cause
- MatGL selection was previously first-come/first-serve in iteration order.

### Intervention
- Added deterministic selection before prediction:
  - define `matgl_needed` candidates (forced or MP-missing),
  - keep only `nsites < matgl_max_atoms`,
  - sort by `material_id`,
  - apply `matgl_max_recalc_entries` cap.
- Added regression test for deterministic selected ID set.

## 11) Candidate-Level Provenance Visibility Gap
### Symptom
- Counters existed in tool-level provenance, but individual candidates lacked explicit source/skipped reasons.

### Intervention
- Added per-candidate tags in `candidate.features`:
  - `band_gap_source`
  - `matgl_forced`
  - `matgl_skipped_reason` (when skipped)
- Added tests validating these tags in forced and skipped scenarios.
