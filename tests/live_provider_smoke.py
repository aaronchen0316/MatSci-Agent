"""One bounded, real provider check for the scheduled CI job."""

import json
import os
import urllib.request


url = os.getenv("MATSCI_BASE_URL", "http://127.0.0.1:8000") + "/discover/full"
payload = {
    "research_goal": "Find AlN semiconductors with known band gaps",
    "constraints": {"top_k": 1, "mp_filters": {"formula": "AlN"}},
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "X-API-Key": os.environ["MATSCI_API_TOKEN"]},
    method="POST",
)
with urllib.request.urlopen(request, timeout=300) as response:
    body = json.load(response)
    assert response.headers["X-Request-ID"] == body["request_id"]

assert body["status"] == "success", body.get("reason_code")
assert body["data_source"] == "materials_project"
assert body["candidates"]
assert any(item["tool_name"] == "mp_retriever" for item in body["provenance"])
print("Live provider check passed: real MP provenance and candidates.")
