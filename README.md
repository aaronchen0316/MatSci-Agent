# MatSci-Agent

Production-style scaffold for an agentic materials discovery loop.

## Features
- Strict Pydantic schemas for API and tool contracts
- Tool interfaces with MP-first and MatGL-aware prediction:
  - `mp_retriever` (live Materials Project + mock fallback)
  - `property_predictor` (MP band gap -> MatGL -> torch-hub -> heuristic fallback)
  - `stability_checker` (mock e-above-hull)
- LangGraph workflow with retry/refine loop when stability fails
- FastAPI endpoint: `POST /discover`
- MLflow logging for each tool/step
- Tests and runnable example

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
python -m venv .venv
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
- `MATSCI_MATGL_MODEL`: exact MatGL band-gap model name to load
- `MATSCI_MATGL_MODEL_CANDIDATES`: comma-separated fallback model names
- `MATSCI_MATGL_RELAX_MODEL`: model used by optional relaxation path

## Run API
```bash
uv run uvicorn matsci_agent.api.main:app --app-dir src --reload
```

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
