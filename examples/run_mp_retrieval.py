import json

from matsci_agent.schemas import DiscoveryConstraints, MPRetrieverInput
from matsci_agent.tools.mp_retriever import MPRetriever


if __name__ == "__main__":
    retriever = MPRetriever()
    payload = MPRetrieverInput(
        research_goal="Find semiconductors without silicon and band gap above 2 eV",
        constraints=DiscoveryConstraints(
            banned_elements=["Si"],
            top_k=5,
        ),
    )

    out = retriever.retrieve(payload)
    print("Provenance:")
    print(json.dumps(out.provenance.model_dump(), indent=2, sort_keys=True))
    print("\nCandidates:")
    if not out.candidates:
        print("(none)")
    else:
        headers = ["material_id", "formula", "mp_band_gap_ev", "nsites", "source"]
        rows = [
            [
                c.material_id,
                c.formula,
                c.features.get("mp_band_gap_ev"),
                c.features.get("nsites"),
                c.source,
            ]
            for c in out.candidates
        ]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
        print(header_line)
        print(sep_line)
        for row in rows:
            print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
