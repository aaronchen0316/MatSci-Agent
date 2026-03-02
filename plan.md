# MatSci-Agent Redevelopment Plan (From Scratch)

## 1) Objective
Rebuild the system for band-gap screening with tighter integration across open-source materials informatics tools.

Target query example:
- "Find semiconductor materials with no Si and band gap higher than 2 eV"

Core policy:
- Use Materials Project (MP) band gap when available.
- Only run MatGL band-gap prediction when MP band gap is missing.
- Add `calculate_matgl` flag to force recalculation via MatGL.
- When recalculating with MatGL, limit to max 10 entries and only structures with `< 50` atoms.

## 2) Required Tooling Stack
- Retrieval + structures: `mp-api`, `pymatgen`
- Graph ML: `matgl`, `torch`
- Optional structure workflow from reference tutorial:
  - M3GNet universal potential relaxation before property prediction (when needed)
- Orchestration/API/logging stay as currently scaffolded:
  - LangGraph, FastAPI, MLflow, Pydantic

Reference used for predictor design:
- MatGL tutorial: [Combining the M3GNet Universal Potential with Property Prediction Models](https://matgl.ai/tutorials%2FCombining%20the%20M3GNet%20Universal%20Potential%20with%20Property%20Prediction%20Models.html)
- GitHub source notebook path provided by user:
  - `materialyzeai/matgl/examples/Combining the M3GNet Universal Potential with Property Prediction Models.ipynb`

## 3) End-to-End Flow (Target)
1. User submits natural-language request.
2. NLP parser extracts:
   - target property (`band_gap_ev`)
   - threshold (`min_band_gap_ev`)
   - element constraints (`banned_elements`, `required_elements`)
   - optional control flag (`calculate_matgl`)
3. MP retriever queries candidates and fetches:
   - material id, composition/formula
   - structure (pymatgen `Structure` or dict)
   - MP band gap if available
4. Property resolver policy:
   - if `calculate_matgl == false`:
     - use MP band gap when present
     - MatGL predict only for missing MP band gap
   - if `calculate_matgl == true`:
     - force MatGL prediction for eligible entries
     - apply limits: at most 10 entries, each with atom count < 50
5. Stability checker + ranking
6. Return JSON report with explicit provenance per candidate

## 4) Data Contract Changes
### Discovery constraints (Pydantic)
- Keep/add:
  - `min_band_gap_ev: float | None`
  - `banned_elements: list[str]`
  - `required_elements: list[str]`
  - `max_energy_above_hull: float`
  - `top_k: int`
- Add:
  - `calculate_matgl: bool = False`
  - `matgl_max_recalc_entries: int = 10` (config-level default acceptable)
  - `matgl_max_atoms: int = 50` (config-level default acceptable)

### Candidate/features
- Ensure each candidate carries:
  - MP band gap (nullable)
  - structure payload usable by pymatgen/matgl
  - atom count (`nsites`)

### Predicted properties
- Keep focus on:
  - `band_gap_ev`
  - `uncertainty`
  - `backend`

## 5) Package-Level Integration Design
### MP Retriever (`mp_retriever.py`)
- Use `mp-api` + `MPRester` for summaries and structure retrieval.
- Return candidates with MP band gap + structure where possible.
- Preserve fallback behavior when API unavailable.

### NLP extractor (new module)
- Add lightweight parser for natural language constraints.
- Map phrases like:
  - "no Si" -> `banned_elements=["Si"]`
  - "band gap higher than 2 eV" -> `min_band_gap_ev=2.0`
  - "force recalculation" -> `calculate_matgl=true`

### Property predictor (`property_predictor.py`)
- Implement policy engine:
  - choose MP band gap vs MatGL band gap prediction per candidate
- MatGL path:
  - use `matgl.load_model(...)` for band-gap model if available
  - optional relaxation path per tutorial before prediction when needed
- Enforce recalc guards:
  - candidate subset capped at 10
  - only candidates with `nsites < 50`
- Emit provenance flags:
  - `band_gap_source`: `materials_project` | `matgl` | `fallback`
  - `matgl_forced`: bool
  - `matgl_skipped_reason`: optional string

## 6) Ranking Policy (Band Gap Objective)
- Filter by:
  - stability threshold
  - `min_band_gap_ev`
- Rank by:
  1. stable first
  2. higher band gap
  3. lower energy above hull
- Score example:
  - `score = band_gap_ev - alpha * energy_above_hull`

## 7) Implementation Phases
### Phase A: Contracts and parser
- Add `calculate_matgl` and recalc limits to schema/config.
- Implement NLP extraction utility + tests.

### Phase B: MP-first property resolution
- Refactor predictor to prefer MP band gap.
- Use MatGL only when MP value missing.
- Add provenance and unit tests.

### Phase C: Forced recalculation policy
- Implement `calculate_matgl=true` path.
- Enforce `<=10` entries and `<50` atoms constraints.
- Add tests for cap + atom-count filter.

### Phase D: MatGL quality path
- Integrate tutorial-informed relax-then-predict flow where needed.
- Add timeout/fallback safeguards.

### Phase E: API and reporting
- Expose parser + policy through `/discover`.
- Ensure returned report includes per-candidate provenance source.

## 8) Test Plan
- NLP extraction tests for common prompt patterns.
- Retriever tests for MP data + missing property cases.
- Predictor tests:
  - MP band gap present -> no MatGL call
  - MP band gap missing -> MatGL call
  - `calculate_matgl=true` -> forced MatGL with caps
  - atom count >= 50 -> skip MatGL
- End-to-end API test with sample prompt:
  - "Find semiconductor materials with no Si and band gap higher than 2 eV"

## 9) Operational/Runtime Notes
- Require env vars:
  - `MP_API_KEY`
- Optional model/runtime dependencies:
  - `matgl`, `torch`, `pymatgen`
- Add clear startup validation for missing optional deps.

## 10) Immediate Next Action (first coding task)
Implement Phase A + B first:
1. Add `calculate_matgl` and recalc limits to schema/config.
2. Implement NLP constraint extraction module.
3. Refactor predictor policy to MP-first, MatGL-if-missing.
4. Add tests for policy decisions.

## 11) Current Progress (2026-03-01)
### Completed
- Phase A completed:
  - Added `calculate_matgl` in schema.
  - Added recalc guard config (`matgl_max_recalc_entries=10`, `matgl_max_atoms=50`).
  - Added NLP parser module and tests.
- Phase B completed:
  - Predictor now uses MP band gap first.
  - MatGL path used only when MP band gap missing, or when `calculate_matgl=true`.
  - Added provenance counters (`used_mp_count`, `used_matgl_count`, `forced_matgl_count`, `matgl_skipped_count`, `fallback_count`).
- API output simplified for user-facing responses:
  - `/discover` now returns only `material_id`, `formula`, `band_gap_ev`.
- Parser provider support:
  - Added Anthropic and OpenAI client support.
  - Added provider auto-mode and fallback logic.
  - Added parser debug snapshot (`get_parser_debug_snapshot`) and safe error logging.

### In Progress
- Phase D mostly implemented:
  - Predictor now attempts MatGL band-gap model loading via `matgl.load_model(...)`.
  - Optional relaxation path is active behind `enable_relaxation` with graceful fallback.
  - Torch-hub and heuristic fallbacks remain for environments without MatGL deps.

### Not Completed
- Full MatGL quality path calibration:
  - Need model pinning/benchmarking and graph-converter-specific tuning for all structure types.
- Phase E enhancements:
  - User-facing output is intentionally minimal; full provenance endpoint/view is still pending if needed.

## 12) Next Priority Tasks
1. Pin and benchmark a production MatGL band-gap model name instead of multi-name fallback probing.
2. Harden MatGL adapter for graph-converter-specific model APIs and add deterministic unit tests around adapter behavior.
3. Add one integration test that exercises:
   - typoed element names,
   - `calculate_matgl=true`,
   - recalc cap + atom-count filter,
   - output shape contract.
4. Add API mode or endpoint to expose full per-candidate provenance (beyond summary output).

## 13) Latest Execution Update (2026-03-01, current pass)
### Completed in this pass
- Predictor selection logic now deterministic for MatGL recalc:
  - Build `matgl_needed` set by policy (forced or MP-missing).
  - Filter by `nsites < matgl_max_atoms`.
  - Apply stable ordering by `material_id` before `matgl_max_recalc_entries` cap.
- Per-candidate provenance tags now emitted in `candidate.features`:
  - `band_gap_source`: `materials_project` | `matgl` | `fallback`
  - `matgl_forced`: bool
  - `matgl_skipped_reason`: `recalc_limit_reached` | `atoms_too_high_or_missing_nsites` (when skipped)
- Workflow wiring updated:
  - `matgl_enable_relaxation` and `matgl_relaxation_max_steps` now flow from config to predictor payload.
- Tests updated:
  - Added deterministic forced-recalc behavior assertion.
  - Added provenance-tag assertions for skipped/reused/fallback paths.

### Remaining highest-priority item
- Implement actual MatGL tutorial-grade relaxation and property pipeline in `_maybe_relax_structure` (currently explicit placeholder).

## 14) Latest Execution Update (2026-03-02)
### Completed in this pass
- Enabled optional MatGL functionality in predictor:
  - `property_predictor` now does MatGL-first band-gap inference (`matgl.load_model`) when available.
  - Added optional MatGL relaxer path (`matgl.ext.ase.Relaxer`) with safe no-op fallback.
  - Preserved torch-hub and heuristic fallback chain for missing runtime dependencies.
- Added install path for MatGL stack:
  - `pyproject.toml` now has `matgl` extra (`matgl`, `torch`, `pymatgen`, `ase`).
- Improved retriever demo readability:
  - `examples/run_mp_retrieval.py` now prints pretty provenance JSON and a candidate table.

### Remaining next step
- Run end-to-end validation in a machine with `uv sync --extra matgl` and verify real MatGL predictions against MP-known band gaps.
