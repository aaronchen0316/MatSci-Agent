from matsci_agent.schemas import DiscoveryConstraints, MPRetrieverInput
from matsci_agent.tools.mp_retriever import MPRetriever


if __name__ == "__main__":
    retriever = MPRetriever()
    payload = MPRetrieverInput(
        research_goal="Find stable alloys with high thermal conductivity without cobalt",
        constraints=DiscoveryConstraints(
            banned_elements=["Co"],
            required_elements=["Al"],
            top_k=5,
        ),
    )

    out = retriever.retrieve(payload)
    print("Provenance:", out.provenance.model_dump())
    print("Candidates:")
    for candidate in out.candidates:
        print(f"- {candidate.material_id}: {candidate.formula} ({candidate.source})")
