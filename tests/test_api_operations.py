import json
import threading

from fastapi.testclient import TestClient

import matsci_agent.api.main as api_main
from matsci_agent.schemas import DiscoveryFullResponse, DiscoveryResponse


client = TestClient(api_main.app)
PAYLOAD = {"research_goal": "Find AlN semiconductor materials"}


class StubWorkflow:
    def __init__(self, status="success", reason_code=None):
        self.status = status
        self.reason_code = reason_code
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return DiscoveryResponse(
            research_goal=request.research_goal,
            constraints=request.constraints,
            status=self.status,
            reason_code=self.reason_code,
            iterations=1,
        )

    def run_full(self, request):
        self.requests.append(request)
        return DiscoveryFullResponse(
            research_goal=request.research_goal,
            constraints=request.constraints,
            status=self.status,
            reason_code=self.reason_code,
            iterations=1,
        )


def test_capacity_is_shared_by_discovery_routes(monkeypatch):
    slots = threading.BoundedSemaphore(2)
    monkeypatch.setattr(api_main, "discovery_slots", slots)
    slots.acquire()
    slots.acquire()
    try:
        for route in ("/discover", "/discover/full"):
            response = client.post(route, json=PAYLOAD)
            assert response.status_code == 429
            assert response.headers["Retry-After"] == "1"
            assert response.headers["X-Request-ID"]
    finally:
        slots.release()
        slots.release()


def test_matgl_requires_operator_and_caller_opt_in(monkeypatch):
    stub = StubWorkflow()
    monkeypatch.setattr(api_main, "workflow_factory", lambda: stub)
    monkeypatch.delenv("MATSCI_ENABLE_MATGL", raising=False)
    request = {**PAYLOAD, "constraints": {"calculate_matgl": True}}
    blocked = client.post("/discover", json=request)
    assert blocked.status_code == 422
    assert blocked.headers["X-Request-ID"]
    assert not stub.requests

    monkeypatch.setenv("MATSCI_ENABLE_MATGL", "1")
    monkeypatch.setattr(api_main, "_matgl_missing", lambda: [])
    allowed = client.post("/discover", json=request)
    assert allowed.status_code == 200
    assert stub.requests[-1].constraints.calculate_matgl is True

    default = client.post("/discover", json=PAYLOAD)
    assert default.status_code == 200
    assert stub.requests[-1].constraints.calculate_matgl is False
    assert "calculate_matgl" in stub.requests[-1].constraints.model_fields_set


def test_readiness_checks_optional_matgl_runtime(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "test-mp")
    monkeypatch.setenv("MATSCI_LLM_API_KEY", "test-llm")
    monkeypatch.setenv("MATSCI_API_TOKEN", "test-api")
    monkeypatch.setenv("MATSCI_REQUIRE_API_TOKEN", "1")
    monkeypatch.setenv("MATSCI_ENABLE_MATGL", "1")
    monkeypatch.setattr(api_main.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(api_main, "_matgl_missing", lambda: ["torch_geometric", "MATSCI_MATGL_MODEL"])
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["missing"] == ["torch_geometric", "MATSCI_MATGL_MODEL"]
    assert response.headers["X-Request-ID"]


def test_matgl_import_check_catches_broken_installed_dependency(monkeypatch):
    imports = []

    def import_module(name):
        imports.append(name)
        if name == "torch_geometric":
            raise OSError("native library unavailable")
        return object()

    monkeypatch.setattr(api_main.importlib, "import_module", import_module)
    api_main._matgl_import_failures.cache_clear()
    try:
        assert api_main._matgl_import_failures() == ("torch_geometric",)
        assert api_main._matgl_import_failures() == ("torch_geometric",)
        assert imports.count("torch_geometric") == 1
    finally:
        api_main._matgl_import_failures.cache_clear()


def test_request_log_records_application_failure_without_payload(monkeypatch):
    stub = StubWorkflow(status="failed", reason_code="mp_service_failed")
    monkeypatch.setattr(api_main, "workflow_factory", lambda: stub)
    records = []
    monkeypatch.setattr(api_main.logger, "info", lambda value: records.append(json.loads(value)))
    response = client.post("/discover", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    event = records[-1]
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["route"] == "/discover"
    assert event["http_status"] == 200
    assert event["workflow_status"] == "failed"
    assert event["reason_code"] == "mp_service_failed"
    assert event["failure"] is True
    assert "research_goal" not in event


def test_request_id_and_failure_log_on_auth_error(monkeypatch):
    monkeypatch.setenv("MATSCI_API_TOKEN", "test-api")
    records = []
    monkeypatch.setattr(api_main.logger, "info", lambda value: records.append(json.loads(value)))
    response = client.post("/discover", json=PAYLOAD)
    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == records[-1]["request_id"]
    assert records[-1]["failure"] is True


def test_unhandled_failure_returns_traced_500(monkeypatch):
    class BrokenWorkflow:
        def run(self, _request):
            raise RuntimeError("provider detail that must stay out of the response")

    monkeypatch.setattr(api_main, "workflow_factory", BrokenWorkflow)
    records = []
    monkeypatch.setattr(api_main.logger, "info", lambda value: records.append(json.loads(value)))
    response = client.post("/discover", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert response.headers["X-Request-ID"] == records[-1]["request_id"]
    assert records[-1]["error_type"] == "RuntimeError"
    assert records[-1]["failure"] is True
