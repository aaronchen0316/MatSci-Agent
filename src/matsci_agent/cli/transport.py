from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from matsci_agent.schemas import (
    CandidateBandGapSummary,
    DiscoveryFullResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    DiscoverySummaryResponse,
    summary_from_result,
)
from matsci_agent.workflow.graph import DiscoveryWorkflow


class CLITransportError(RuntimeError):
    """Raised when CLI transport cannot reach or decode backend responses."""


def run_summary(
    request: DiscoveryRequest,
    api_url: str | None = None,
    enable_policy_filter: bool | None = None,
) -> DiscoverySummaryResponse:
    if api_url:
        return _run_summary_http(request, api_url)
    workflow = DiscoveryWorkflow(enable_policy_filter=enable_policy_filter)
    result = workflow.run(request)
    return _build_summary_response(result)


def run_full(
    request: DiscoveryRequest,
    api_url: str | None = None,
    enable_policy_filter: bool | None = None,
) -> DiscoveryFullResponse:
    if api_url:
        return _run_full_http(request, api_url)
    workflow = DiscoveryWorkflow(enable_policy_filter=enable_policy_filter)
    return workflow.run_full(request)


def probe_health(api_url: str) -> tuple[bool, str]:
    url = _join_url(api_url, "/health")
    try:
        payload = _post_json(url=None, body=None, method="GET", full_url=url)
    except CLITransportError as exc:
        return False, str(exc)
    if payload.get("status") != "ok":
        return False, f"Unexpected health payload: {payload!r}"
    return True, "Server responded with status=ok."


def _run_summary_http(request: DiscoveryRequest, api_url: str) -> DiscoverySummaryResponse:
    payload = _post_json(
        _join_url(api_url, "/discover"),
        request.model_dump(mode="json"),
    )
    try:
        return DiscoverySummaryResponse.model_validate(payload)
    except Exception as exc:  # pragma: no cover - exercised by transport failure path
        raise CLITransportError(f"Invalid /discover response: {exc}") from exc


def _run_full_http(request: DiscoveryRequest, api_url: str) -> DiscoveryFullResponse:
    payload = _post_json(
        _join_url(api_url, "/discover/full"),
        request.model_dump(mode="json"),
    )
    try:
        return DiscoveryFullResponse.model_validate(payload)
    except Exception as exc:  # pragma: no cover - exercised by transport failure path
        raise CLITransportError(f"Invalid /discover/full response: {exc}") from exc


def _build_summary_response(result: DiscoveryResponse) -> DiscoverySummaryResponse:
    return summary_from_result(result)


def _join_url(api_url: str, path: str) -> str:
    return f"{api_url.rstrip('/')}{path}"


def _post_json(
    url: str | None,
    body: dict | None,
    method: str = "POST",
    full_url: str | None = None,
) -> dict:
    target = full_url or url
    if target is None:
        raise CLITransportError("Missing HTTP target URL.")

    request_data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if os.getenv("MATSCI_API_TOKEN"):
        headers["X-API-Key"] = os.environ["MATSCI_API_TOKEN"]
    request = urllib.request.Request(
        target,
        data=request_data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CLITransportError(f"HTTP {exc.code} from {target}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise CLITransportError(f"Cannot reach {target}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise CLITransportError(f"Timeout reaching {target}.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CLITransportError(f"Invalid JSON from {target}: {exc}") from exc
