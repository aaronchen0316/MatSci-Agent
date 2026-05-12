from fastapi import FastAPI

from matsci_agent.config import settings
from matsci_agent.schemas import (
    CandidateBandGapSummary,
    DiscoveryFullResponse,
    DiscoveryRequest,
    DiscoverySummaryResponse,
)
from matsci_agent.workflow.graph import DiscoveryWorkflow

app = FastAPI(title=settings.app_name, version="0.1.0")
workflow = DiscoveryWorkflow()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/discover", response_model=DiscoverySummaryResponse)
def discover(payload: DiscoveryRequest) -> DiscoverySummaryResponse:
    result = workflow.run(payload)
    return DiscoverySummaryResponse(
        status=result.status,
        candidates=[
            CandidateBandGapSummary(
                material_id=rc.candidate.material_id,
                formula=rc.candidate.formula,
                band_gap_ev=rc.predicted_properties.band_gap_ev,
            )
            for rc in result.candidates
        ],
        messages=result.messages,
        unsupported_reason=(
            result.capability_assessment.reason_message
            if result.capability_assessment is not None
            and not result.capability_assessment.supported
            else None
        ),
    )


@app.post("/discover/full", response_model=DiscoveryFullResponse)
def discover_full(payload: DiscoveryRequest) -> DiscoveryFullResponse:
    return workflow.run_full(payload)
