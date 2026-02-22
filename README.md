# MatSci-Agent

Production-style scaffold for an agentic materials discovery loop.

## Features
- Strict Pydantic schemas for API and tool contracts
- Tool interfaces with mock backends:
  - `mp_retriever` (Materials Project stub)
  - `property_predictor` (pluggable backend, mock default)
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
pip install -e .[dev]
```

## Run API
```bash
uvicorn matsci_agent.api.main:app --app-dir src --reload
```

## Example Request
```bash
curl -X POST http://127.0.0.1:8000/discover \
  -H "Content-Type: application/json" \
  -d '{
    "research_goal": "Find a stable alloy with high thermal conductivity that does not use Cobalt",
    "constraints": {
      "banned_elements": ["Co"],
      "max_energy_above_hull": 0.08,
      "top_k": 5
    }
  }'
```

## Run Example Script
```bash
python examples/run_discovery.py
```

## Run Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t matsci-agent:local .
docker run --rm -p 8000:8000 matsci-agent:local
```

## TODOs for Real Integrations
- Replace `MPRetriever.retrieve` mock with authenticated Materials Project API calls.
- Replace `MockCompositionBackend` with CrabNet/Roost inference service.
- Add stage-2 structure model refinement using CHGNet/M3GNet/ALIGNN.
- Add calibrated uncertainty + DFT triage queue.
- Enrich provenance with exact MP query IDs and model artifact versions.
