# MatSci-Agent

MatSci-Agent is a research prototype for screening Materials Project entries from a natural language request. It returns a short list with material IDs, band gaps, stability annotations, and available Materials Project properties. It supports band gap screening and screening by Materials Project summary fields such as formation energy, density, and volume.

## How it works

1. Parse the request into typed constraints and reject tasks outside the supported scope.
2. Use an LLM to propose a bounded set of formulas, then search Materials Project.
3. Use an LLM chemistry screen to filter candidates. The filter is enabled by default.
4. Apply requested limits to each MP structure, select up to `top_k` formulas, and report every passing structure for those formulas. Optional MatGL calculation runs only when requested.

The workflow uses `gpt-5.5` by default through an OpenAI compatible API. Search expansion and chemistry screening need a working LLM connection. If either step fails, the request can return a failed result rather than an unfiltered list.

## Quick start

From the repository root, with `uv` installed:

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra mp
export OPENROUTER_API_KEY="<your LLM key>"
export MP_API_KEY="<your Materials Project key>"
```

The default LLM endpoint is OpenRouter. Set `MATSCI_LLM_BASE_URL`, `MATSCI_LLM_MODEL`, and `MATSCI_LLM_API_KEY` to select an endpoint, model, and key. `MATSCI_LLM_MODEL` takes precedence over the older `MATSCI_NLP_MODEL`. `MATSCI_API_KEY` remains a compatibility alias; `MATSCI_LLM_API_KEY` wins when both are set. If your key uses another environment variable, set `MATSCI_LLM_API_KEY_ENV` to its name. `MATSCI_API_TOKEN` is separate: it protects the HTTP API.

Live discovery requires `MP_API_KEY` and the `mp` extra. Missing credentials, MP errors, and searches with no matches have separate reason codes. User facing commands and API routes never substitute mock candidates. Keep keys out of source control.

## Use the CLI

```bash
uv run matsci doctor
uv run matsci demo "Find semiconductors without silicon and band gap above 2 eV"
uv run matsci operator "Find semiconductor materials"
uv run matsci examples list
uv run matsci examples run basic_success
```

`matsci demo` shows one row per formula and marks formulas with multiple structures. `matsci operator` shows every structure and the full workflow trace. Commands run locally unless you pass `--api-url http://127.0.0.1:8000`. The examples use preset live requests; `matsci doctor` checks configuration and dependencies.

## Use the API

Start the server:

```bash
uv run uvicorn matsci_agent.api.main:app --app-dir src --reload
```

Send a request:

```bash
curl -X POST http://127.0.0.1:8000/discover \
  -H "Content-Type: application/json" \
  -d '{
    "research_goal": "Find semiconductors without silicon and band gap above 2 eV",
    "constraints": {
      "banned_elements": ["Si"],
      "min_band_gap_ev": 2.0,
      "top_k": 5
    }
  }'
```

Both `POST /discover` and `POST /discover/full` return every passing structure for the selected formulas. The full route also includes the plan, formula targets, filter decisions, and provenance. Responses include `status`, `reason_code`, `data_source`, and `request_id`; the request ID is also in the `X-Request-ID` header. A successful search with no matches has `status: success`, `reason_code: no_matches`, and an empty candidate list. `GET /health` checks the process; `GET /ready` checks required credentials and the MP client.

For detailed Materials Project queries, use `constraints.mp_filters`. It accepts fields such as `formula`, `elements`, `formation_energy`, `density`, `volume`, and `crystal_system`. Numeric ranges use `{"min": value, "max": value}`. Set `max_energy_above_hull` only when you want to filter retrieval by that value; otherwise stability is only annotated.

## Code map

`src/matsci_agent/api/main.py` exposes the API. `src/matsci_agent/workflow/graph.py` runs the workflow. `src/matsci_agent/schemas.py` defines request and result types. The parser and search expansion code live in `nlp/` and `agents/`; retrieval, filtering, and prediction live in `tools/`. Add regression tests in `tests/`.

To run tests, install the development extra with `uv sync --extra dev --extra mp`, then run `uv run pytest -q`.

## Optional MatGL calculation

Install the extra and request `"calculate_matgl": true` in `constraints`, or pass `--calculate-matgl` to `matsci demo`. MP limits and chemistry screening select entries first. MatGL reports a value or a failure for each selected structure. The MP and MatGL gaps remain separate, and their numeric comparison is labeled `gap_difference_ev`. A MatGL result below a requested MP threshold remains visible. If more than 10 structures are selected, the calculation stops before it starts.

```bash
uv sync --extra mp --extra matgl
```

The default model paths are `models/pretrained/MEGNet-BandGap-mfi-MP-2019.4.1` for band gaps and `models/pretrained/TensorNet-PES-MatPES-PBE-2025.2` for optional relaxation. Both bundles come from the Materialyze model registry and run through PyG. The multifidelity band-gap model uses state 0 (PBE), matching the MP comparison. Set `MATSCI_MATGL_MODEL` or `MATSCI_MATGL_RELAX_MODEL` to override them. The `matgl` extra locks MatGL 4.1.1 and CPU PyTorch 2.13.0; DGL is absent. Relaxation is disabled by default.

## Current stage and limits

This is a bounded screening prototype, not a validated discovery platform. It supports Materials Project summary properties, band gap screening, and optional local MatGL calculation. It does not run molecular dynamics, diffusivity, transport, defect, or general simulation workflows.

Stability uses Materials Project energy above hull when present and is unknown when missing. A missing MP band gap is unknown unless a requested band gap limit requires excluding that entry. No gap is inferred from the formula or substituted when MatGL fails. `top_k` can be 1 through 30 formulas; the number of returned structures may be larger.

## Container deployment

The container uses digest-pinned base images, `uv.lock`, and bundled model files. Build the standard MP and LLM image with `docker build -t matsci-agent .`. The optional Linux MatGL image uses `docker build --build-arg WITH_MATGL=1 -t matsci-agent:matgl .`; it installs CPU-only PyTorch and PyG. Supply `MP_API_KEY`, the LLM key, and `MATSCI_API_TOKEN` at runtime. The container requires the API token on discovery routes; clients send it as `X-API-Key`. For example:

```bash
docker run --rm -p 8000:8000 --env-file .env matsci-agent
```

Use one worker. To allow MatGL requests in the MatGL image, set `MATSCI_ENABLE_MATGL=1` at runtime; callers must also set `constraints.calculate_matgl=true`. `/health` checks the process; `/ready` checks credentials, the MP client, and enabled MatGL imports and local files without contacting providers or loading models on every probe. The API accepts at most two concurrent discovery requests and returns `429` with `Retry-After` when full. MP retrieval screens at most 120 distinct documents across at most three one-page searches. Keep a per-client rate limit at the front server.

Every response has `X-Request-ID`. One JSON log line per request records that ID, route, duration, HTTP and workflow status, reason code, and a failure flag. It omits request text and credentials; count `failure=true` events for operational failure metrics. CI builds and smokes the standard image on every change. A weekly or manual run uses real `MP_API_KEY` and `OPENROUTER_API_KEY` GitHub secrets for one end-to-end request and loads both models in the MatGL image. A Linux container build has not been run on this local machine because Docker is unavailable here.
