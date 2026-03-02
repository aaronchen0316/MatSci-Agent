from matsci_agent.schemas import DiscoveryConstraints, DiscoveryRequest
from matsci_agent.workflow.graph import DiscoveryWorkflow


if __name__ == "__main__":
    workflow = DiscoveryWorkflow()
    request = DiscoveryRequest(
        research_goal="Find stable semiconductors without silicon and band gap above 2 eV",
        constraints=DiscoveryConstraints(
            banned_elements=["Si"],
            min_band_gap_ev=2.0,
            max_energy_above_hull=0.08,
            top_k=3,
        ),
    )
    response = workflow.run(request)
    print(response.model_dump_json(indent=2))
