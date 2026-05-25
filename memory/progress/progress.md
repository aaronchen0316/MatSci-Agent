# MatSci-Agent Progress

## Main Objective
Build an MVP-first materials query system that accepts natural-language research goals, retrieves candidate bulk materials from Materials Project, and returns compact user-facing output through `POST /discover`, while keeping richer screening/reporting systems in repo as optional non-MVP layers.

## Current State
- `POST /discover` already provides natural-language query -> compact result output.
- `POST /discover` now returns compact annotated shortlist items including band-gap source and stability fields.
- `POST /discover/full` exists as a debug/audit surface, not a required normal-user endpoint.
- `ChemistryIntentAgent` turns request text into typed planning state for downstream execution.
- `CapabilityGuardrail` already bounds scope and refuses unsupported task classes.
- `MPRetriever` handles Materials Project lookup or mock fallback.
- `policy_filter` remains available as optional LLM-based post-retrieval chemistry screening and is disabled by default unless explicitly enabled.
- `PropertyPredictor`, ranking, and reporting layers exist, but repo scope already exceeds narrow MVP needs.
- `StabilityChecker` uses MP `energy_above_hull` when available and otherwise returns honest `stability_unknown`.
- MLflow observability is wired in, but not required for basic MVP behavior.
- Offline benchmark tooling exists, and a provisional small baseline artifact set is now published for optional prediction/recalc regression tracking.

## Biggest Current Limitation
- Biggest current limitation is product-scope mismatch.
- Repo already contains broader screening-platform pieces than near-term MVP requires.
- Near-term work should reduce ambiguity around primary user flow before expanding scientific/platform depth.

## MVP Blockers
1. Simplify and validate `POST /discover` as primary user contract.
2. Confirm Materials Project retrieval quality for actual user query patterns.
3. Decide whether local prediction/recalc remains part of MVP or becomes secondary.
4. Keep optional chemistry filtering only where it clearly improves result quality for MVP use cases.

## Non-MVP Future Work
1. Publish benchmark baselines if local prediction/recalc remains product-critical.
2. Broaden chemistry filtering beyond current `band_gap_screening` scope and compact metadata.
3. Expand planning into richer scientific ontology if product scope broadens.
4. Add richer external-facing reporting beyond raw `/discover/full` trace if compact output proves too thin.
5. Expand supported task classes only when real execution paths are added.

## Bottom Line
Repository already contains a capable materials-screening scaffold, but near-term product target is narrower than current architecture.
MVP does not require benchmark publication, richer planning ontology, broader task coverage, or MLflow-heavy workflows.
Priority is clarifying and tightening retrieval-first user flow through `POST /discover`, while keeping advanced systems available as optional layers.
