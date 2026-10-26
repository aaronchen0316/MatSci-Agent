import importlib
import json
import logging
import os
import secrets
import threading
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from matsci_agent.config import settings
from matsci_agent.nlp.parser import resolve_llm_api_key
from matsci_agent.observability.mlflow_logger import MLflowLogger
from matsci_agent.schemas import (
    DiscoveryFullResponse,
    DiscoveryRequest,
    DiscoverySummaryResponse,
    summary_from_result,
)
from matsci_agent.workflow.graph import DiscoveryWorkflow

app = FastAPI(title=settings.app_name, version="0.1.0")
logger = logging.getLogger("matsci_agent.requests")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False
discovery_slots = threading.BoundedSemaphore(2)
mlflow_logger = MLflowLogger(settings.mlflow_experiment)


def workflow_factory() -> DiscoveryWorkflow:
    return DiscoveryWorkflow(logger=mlflow_logger)


@app.middleware("http")
async def request_trace(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    started = time.monotonic()
    error_type = None
    try:
        response = await call_next(request)
    except Exception as exc:
        error_type = type(exc).__name__
        response = JSONResponse(status_code=500, content={"detail": "Internal server error."})
    response.headers["X-Request-ID"] = request_id
    workflow_status = getattr(request.state, "workflow_status", None)
    event = {
        "event": "request_completed",
        "request_id": request_id,
        "route": request.scope.get("route").path if request.scope.get("route") else request.url.path,
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "http_status": response.status_code,
        "workflow_status": workflow_status,
        "reason_code": getattr(request.state, "reason_code", None),
        "failure": response.status_code >= 400 or workflow_status in {"failed", "partial"},
    }
    if error_type:
        event["error_type"] = error_type
    logger.info(json.dumps(event, separators=(",", ":")))
    return response


def _authorize(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("MATSCI_API_TOKEN")
    if not expected and os.getenv("MATSCI_REQUIRE_API_TOKEN") == "1":
        raise HTTPException(status_code=503, detail="API token is not configured.")
    if expected and (x_api_key is None or not secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid API token.")


@lru_cache(maxsize=1)
def _matgl_import_failures() -> tuple[str, ...]:
    missing = []
    for name in ("torch", "torch_geometric", "matgl", "ase"):
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return tuple(missing)


def _matgl_missing() -> list[str]:
    missing = list(_matgl_import_failures())
    for variable, default in (
        ("MATSCI_MATGL_MODEL", "models/pretrained/MEGNet-BandGap-mfi-MP-2019.4.1"),
        ("MATSCI_MATGL_RELAX_MODEL", "models/pretrained/TensorNet-PES-MatPES-PBE-2025.2"),
    ):
        model_dir = Path(os.getenv(variable, default))
        if not all((model_dir / name).is_file() for name in ("model.json", "model.pt", "state.pt")):
            missing.append(variable)
    return missing


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready():
    missing = []
    if not os.getenv("MP_API_KEY"):
        missing.append("MP_API_KEY")
    llm_key, llm_key_env = resolve_llm_api_key()
    if not llm_key:
        missing.append(llm_key_env)
    if importlib.util.find_spec("mp_api") is None:
        missing.append("mp-api")
    if os.getenv("MATSCI_REQUIRE_API_TOKEN") == "1" and not os.getenv("MATSCI_API_TOKEN"):
        missing.append("MATSCI_API_TOKEN")
    if os.getenv("MATSCI_ENABLE_MATGL") == "1":
        missing.extend(_matgl_missing())
    status = 200 if not missing else 503
    return JSONResponse(status_code=status, content={"status": "ready" if not missing else "not_ready", "missing": missing})


def _run_discovery(payload: DiscoveryRequest, request: Request, *, full: bool):
    requested_matgl = payload.constraints.calculate_matgl
    if requested_matgl and os.getenv("MATSCI_ENABLE_MATGL") != "1":
        raise HTTPException(status_code=422, detail="MatGL calculations are disabled on this server.")
    if requested_matgl and _matgl_missing():
        raise HTTPException(status_code=503, detail="MatGL runtime is unavailable.")
    if not discovery_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Discovery capacity reached.", headers={"Retry-After": "1"})
    try:
        # Keep MatGL an explicit request choice; the parser must not turn it on.
        constraints = payload.constraints.model_copy(deep=True)
        constraints.calculate_matgl = requested_matgl
        bounded_payload = payload.model_copy(update={"constraints": constraints})
        runner = workflow_factory()
        result = runner.run_full(bounded_payload) if full else runner.run(bounded_payload)
        result.request_id = request.state.request_id
        request.state.workflow_status = result.status
        request.state.reason_code = result.reason_code
        return result if full else summary_from_result(result)
    finally:
        discovery_slots.release()


@app.post("/discover", response_model=DiscoverySummaryResponse)
def discover(payload: DiscoveryRequest, request: Request, _: None = Depends(_authorize)) -> DiscoverySummaryResponse:
    return _run_discovery(payload, request, full=False)


@app.post("/discover/full", response_model=DiscoveryFullResponse)
def discover_full(payload: DiscoveryRequest, request: Request, _: None = Depends(_authorize)) -> DiscoveryFullResponse:
    return _run_discovery(payload, request, full=True)
