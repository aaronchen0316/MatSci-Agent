from fastapi.testclient import TestClient

from matsci_agent.api.main import app
from matsci_agent.schemas import DiscoveryConstraints, DiscoveryRequest
from matsci_agent.workflow.graph import DiscoveryWorkflow


client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_discover_endpoint_returns_ranked_candidates():
    payload = {
        "research_goal": "Find semiconductor materials without silicon and band gap above 2 eV",
        "constraints": {
            "banned_elements": ["Si"],
            "min_band_gap_ev": 2.0,
            "max_energy_above_hull": 0.08,
            "top_k": 3,
        },
    }
    res = client.post("/discover", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"success", "partial", "failed"}
    assert len(body["candidates"]) <= 3
    if body["candidates"]:
        first = body["candidates"][0]
        assert set(first.keys()) == {"material_id", "formula", "band_gap_ev"}


def test_workflow_runs_with_retry_cap():
    wf = DiscoveryWorkflow()
    req = DiscoveryRequest(
        research_goal="Find high band gap materials",
        constraints=DiscoveryConstraints(max_energy_above_hull=0.0, top_k=2),
    )
    out = wf.run(req)
    assert out.iterations >= 1
    assert len(out.candidates) <= 2


def test_discover_endpoint_refuses_unsupported_diffusivity_request():
    payload = {
        "research_goal": "Estimate diffusivity in bulk materials with long molecular dynamics runs",
    }
    res = client.post("/discover", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "unsupported"
    assert body["candidates"] == []
    assert "unsupported" in body["unsupported_reason"].lower()
