from fastapi import FastAPI

from matsci_agent.config import settings
from matsci_agent.schemas import DiscoveryRequest, DiscoveryResponse
from matsci_agent.workflow.graph import DiscoveryWorkflow

app = FastAPI(title=settings.app_name, version="0.1.0")
workflow = DiscoveryWorkflow()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/discover", response_model=DiscoveryResponse)
def discover(payload: DiscoveryRequest) -> DiscoveryResponse:
    return workflow.run(payload)
