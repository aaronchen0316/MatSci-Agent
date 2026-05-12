# MatSci-Agent

Production-style scaffold for an agentic materials discovery loop.

## Features
- Strict Pydantic schemas for API and tool contracts
- Tool interfaces with MP-first and MatGL-aware prediction:
  - `mp_retriever` (live Materials Project + mock fallback)
  - `property_predictor` (MP band gap -> MatGL -> torch-hub -> heuristic fallback)
  - `stability_checker` (MP `energy_above_hull` -> stability unknown when MP hull data is missing)
  - `policy_filter` (LLM-backed chemistry filtering with strict validation and fail-closed behavior)
- LangGraph workflow with planning, capability guardrail, chemistry filtering, and retry/refine loop when stability fails
- FastAPI endpoint: `POST /discover`
- FastAPI debug trace endpoint: `POST /discover/full`
- MLflow logging for each tool/step
- Tests, runnable examples, and offline benchmark script

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
- `POST /discover`: compact user-facing shortlist
- `POST /discover/full`: debug/audit trace with internal workflow artifacts

If the LLM chemistry filter is enabled through normal runtime, the same provider credentials used by the parser are also required here:
- `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY`
- or `OPENAI_API_KEY`

The chemistry filter:
- only applies to `band_gap_screening`
- fails closed on timeout, invalid JSON, or incomplete candidate decisions
- uses one bounded replenish pass when the first kept set underfills `top_k`
- appears in full detail through `/discover/full`, including filter decisions and provenance

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

## Run Tests
```bash
uv run pytest -q
```

## Docker
```bash
docker build -t matsci-agent:local .
docker run --rm -p 8000:8000 matsci-agent:local
```

## TODOs for Real Integrations
- Improve model-specific MatGL adapters for graph conversion edge cases.
- Add calibrated uncertainty + DFT triage queue.
- Enrich provenance with exact MP query IDs and model artifact versions.
- Broaden the chemistry filter beyond current practical vs exploratory screening.

## Current Limitations
- Stability is intentionally conservative:
  - MP `energy_above_hull` is used when available.
  - when MP hull data is missing, candidates are returned as stability unknown.
  - no local proxy or MatGL-based stability estimate is used.
- Chemistry filter only applies to `band_gap_screening`.
- Chemistry filter reasons from compact candidate metadata, not richer structure-aware chemistry features.
- Benchmark tooling exists, but benchmark results are not yet recorded in repo artifacts/docs.
- Planning layer is still narrow: parser output plus deterministic enrichment, not a richer chemistry ontology.
- Reporting is compact and deterministic, not a richer scientific analysis layer.

## Current Assessment
- Good engineering scaffold for agentic bulk-inorganic band-gap screening.
- Strong control-plane pieces already exist:
  - typed planning
  - deterministic capability guardrail
  - LLM chemistry filtering with strict validation
  - bounded local model execution
  - offline benchmark tooling
- Not yet a research-grade materials discovery system because chemistry coverage is narrow and benchmark results are not yet published.
