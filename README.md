# MatSci-Agent

Production-style scaffold for retrieval-first Materials Project property screening over bounded candidate search spaces.

## Features
- Strict Pydantic schemas for API and tool contracts
- Tool interfaces with MP-first retrieval and optional MatGL-aware prediction:
  - `mp_retriever` (live Materials Project + mock fallback)
  - `property_predictor` (MP band gap -> MatGL -> torch-hub -> heuristic fallback)
  - `stability_checker` (MP `energy_above_hull` annotation -> stability unknown when MP hull data is missing)
- `search_space_expander` (fail-closed OpenRouter-backed formula-target expansion before retrieval)
- `policy_filter` (default-on single LLM chemistry screen after retrieval with deterministic hard reject guardrail for impractical elements)
- LangGraph workflow with planning, capability guardrail, search-space expansion, retrieval, default-on chemistry filtering, MP-only generic property summaries, optional band-gap prediction, stability annotation, ranking, and reporting
- FastAPI endpoint: `POST /discover`
- FastAPI debug trace endpoint: `POST /discover/full`
- Optional MLflow logging for each tool/step
- Tests, runnable examples, and offline benchmark script/artifacts

## Project Structure
- `src/matsci_agent/schemas.py`: API + tool I/O schemas
- `src/matsci_agent/tools/`: tool interface implementations
- `src/matsci_agent/workflow/graph.py`: LangGraph orchestration
- `src/matsci_agent/api/main.py`: FastAPI app
- `src/matsci_agent/observability/mlflow_logger.py`: MLflow wrapper
- `examples/run_discovery.py`: local example run
- `tests/test_discover.py`: baseline tests

## Setup
```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --extra dev
```

## Optional: Enable Live Materials Project Retrieval
```bash
uv sync --extra dev --extra mp
export MP_API_KEY="<your-key>"
```

Keep your key in env vars (or a local `.env` that is gitignored), not in source code.

## Optional: Enable MatGL Band-Gap Prediction
```bash
uv sync --extra dev --extra matgl
```

Optional runtime env vars:
- `MATSCI_MATGL_MODEL`: exact MatGL band-gap model name or local model directory (default: `models/pretrained/MEGNet-MP-2019.4.1-BandGap-mfi`)
- `MATSCI_MATGL_MODEL_CANDIDATES`: comma-separated fallback model names
- `MATSCI_MATGL_RELAX_MODEL`: model name or local directory used by the optional relaxation path (default: `models/pretrained/TensorNet-PES-MatPES-PBE-2025.2`)

Note:
- The default band-gap model now points to the local downloaded bundle at `models/pretrained/MEGNet-MP-2019.4.1-BandGap-mfi`.
- `MEGNet-MP-2019.4.1-BandGap-mfi` is DGL-backed, so Python `3.12` is the recommended runtime.
- This scaffold includes a compatibility inference path for this legacy MEGNet checkpoint on modern MatGL runtimes.
- The optional relaxation path now defaults to the local downloaded bundle at `models/pretrained/TensorNet-PES-MatPES-PBE-2025.2`.

## Run API
```bash
uv run uvicorn matsci_agent.api.main:app --app-dir src --reload
```

Endpoint roles:
- `POST /discover`: compact annotated shortlist for normal user flow
- `POST /discover/full`: debug/audit trace with internal workflow artifacts

For parser + search-space expansion + policy-filter LLM calls, configure OpenRouter:
- `OPENROUTER_API_KEY_RAG` or `OPENROUTER_API_KEY`
- optional `MATSCI_LLM_API_KEY_ENV` to point at a different env var name
- optional `MATSCI_LLM_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- optional `MATSCI_NLP_MODEL` / `MATSCI_LLM_MODEL` / `MATSCI_OPENROUTER_MODEL`
  default model: `openai/gpt-oss-120b:free`

The search-space expander:
- runs before retrieval for supported screening requests
- returns bounded formula targets with normalized formula, `chemsys`, confidence, and rationale
- defaults to `min(max(top_k * 3, top_k), 30)` targets
- fails closed if OpenRouter is unavailable or no valid MP-compatible formula targets remain
- appears in `/discover/full` as `search_space_targets` plus provenance

The chemistry filter:
- applies to `band_gap_screening` and `mp_property_screening`
- is enabled by default after retrieval
- uses one policy name: `chemistry_screening`
- sees `source_universe` and parser-derived `requested_material_class` in its discovery context
- fails closed when no remote LLM provider credentials are available
- fails closed on timeout, invalid JSON, or incomplete candidate decisions
- always rejects candidates containing impractical/radioactive elements
- uses one bounded replenish pass when the first kept set underfills `top_k`
- appears in full detail through `/discover/full`, including filter decisions and provenance

`DiscoveryConstraints` now supports nested typed `mp_filters` for richer `mpr.materials.summary.search(...)` intent, while keeping existing top-level aliases for compatibility:
- `banned_elements` -> `exclude_elements`
- `required_elements` -> `elements`
- `min_band_gap_ev` -> `mp_filters.band_gap.min`
- `max_energy_above_hull` -> `mp_filters.energy_above_hull.max`

`POST /discover` candidate summaries include:
- `material_id`
- `formula`
- `band_gap_ev`
- `band_gap_source`
- `energy_above_hull`
- `is_stable`
- `stability_source`
- `has_multiple_entries`
- `entry_count`
- `properties` for generic MP fields such as `formation_energy`, `density`, `volume`, and MP summary annotations

Retrieval deduplicates exact `formula_pretty` values from Materials Project:
- highest-ranked representative per formula is kept
- duplicate metadata reports whether multiple MP entries existed for that formula

`DiscoveryPlan` now separates:
- `source_universe`: deterministic backend/source truth such as `materials_project_entries`
- `requested_material_class`: parser-derived user-intent class such as `semiconductor`, `amine_solvent`, or `unknown`

## Example Request
```bash
curl -X POST http://127.0.0.1:8000/discover \
  -H "Content-Type: application/json" \
  -d '{
    "research_goal": "Find semiconductor materials with no Si and band gap higher than 2 eV",
    "constraints": {
      "banned_elements": ["Si"],
      "min_band_gap_ev": 2.0,
      "max_energy_above_hull": 0.08,
      "top_k": 5
    }
  }'
```

## Run Example Scripts
```bash
uv run python examples/run_discovery.py
uv run python examples/run_mp_retrieval.py
uv run python examples/benchmark_bandgap_predictor.py --mode small --output artifacts/bandgap_benchmark.json
```

## Published Small Baseline
- Artifact files:
  - `artifacts/bandgap_benchmark_small.json`
  - `artifacts/bandgap_benchmark_small.csv`
  - `artifacts/bandgap_benchmark_small.md`
- Generated: `2026-05-25T17:47:20.297358Z`
- Metrics:
  - `sample_size=10`
  - `mae=1.6451`
  - `rmse=2.2116`
  - `rank_correlation=0.7178`
  - `matgl_count=10`
  - `fallback_count=0`

This is a provisional small-mode baseline for the optional prediction/recalc path. It measures agreement against MP-known band gaps, not absolute experimental truth.

## Run Tests
```bash
uv run pytest -q
```

## CLI
```bash
uv run matsci demo "Find semiconductor materials without silicon and band gap above 2 eV"
uv run matsci operator "Find semiconductor materials"
uv run matsci doctor
uv run matsci scenarios list
uv run matsci scenarios run basic_success
```

CLI modes:
- `matsci demo`: polished single-shot shortlist view
- `matsci operator`: full trace/debug console over workflow output
- `matsci doctor`: read-only environment diagnostics
- `matsci scenarios list|run`: built-in demo presets

By default the CLI runs the workflow in-process. Add `--api-url http://127.0.0.1:8000` to use the live HTTP API instead.

## Docker
```bash
docker build -t matsci-agent:local .
docker run --rm -p 8000:8000 matsci-agent:local
```

## TODOs for Real Integrations
- Improve model-specific MatGL adapters for graph conversion edge cases.
- Add calibrated uncertainty + DFT triage queue.
- Enrich provenance with exact MP query IDs and model artifact versions.
- Add richer class-aware expansion/evaluation data beyond current formula-level OpenRouter expansion and compact metadata.

## Current Limitations
- Stability is intentionally conservative:
  - MP `energy_above_hull` is used when available.
  - when MP hull data is missing, candidates are returned as stability unknown.
  - no local proxy or MatGL-based stability estimate is used.
- Search-space expansion is fail-closed and depends on remote LLM credentials for supported screening runs.
- Generic MP-property screening is MP-only and does not run MatGL property prediction.
- Chemistry filter is default-on for `band_gap_screening` and `mp_property_screening`, and still reasons from compact metadata.
- Chemistry filter reasons from compact candidate metadata, not richer structure-aware chemistry features.
- Parser now supports a richer typed `mp_filters` surface, but many Materials Project search kwargs are still intentionally out of v1 scope.
- Benchmark baseline currently measures optional prediction/recalc quality against MP-known band gaps, not absolute experimental truth.
- The committed baseline is small and provisional; it is useful for regression tracking, not broad scientific validation.
- Planning layer is still narrow: parser output plus deterministic enrichment, not a richer chemistry ontology.
- Reporting is compact and deterministic, not a richer scientific analysis layer.

## Current Assessment
- Good engineering scaffold for retrieval-first Materials Project property screening.
- Strong control-plane pieces already exist:
  - typed planning
  - deterministic capability guardrail
  - default-on LLM chemistry filtering with strict validation
  - bounded local model execution
  - offline benchmark tooling
- Not yet a research-grade materials discovery system because chemistry coverage is narrow and planning/reporting remain intentionally thin.
