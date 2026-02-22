from matsci_agent.schemas import DiscoveryConstraints, DiscoveryRequest
from matsci_agent.workflow.graph import DiscoveryWorkflow


if __name__ == "__main__":
    workflow = DiscoveryWorkflow()
    request = DiscoveryRequest(
        research_goal="Find a stable alloy with high thermal conductivity without cobalt",
        constraints=DiscoveryConstraints(
            banned_elements=["Co"],
            min_thermal_conductivity=120,
            max_energy_above_hull=0.08,
            top_k=3,
        ),
    )
    response = workflow.run(request)
    print(response.model_dump_json(indent=2))
