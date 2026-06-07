# MatSci-Agent Progress

## Main Objective
Build an MVP-first materials query system that accepts natural-language research goals, retrieves candidate bulk materials from Materials Project, and returns compact user-facing output through `POST /discover`, while keeping richer screening/reporting systems in repo as optional non-MVP layers.

## Current State
- `POST /discover` already provides natural-language query -> compact result output.
- `POST /discover` now returns compact annotated shortlist items including band-gap source and stability fields.
- `POST /discover/full` exists as a debug/audit surface, not a required normal-user endpoint.
- `ChemistryIntentAgent` turns request text into typed planning state for downstream execution.
- `CapabilityGuardrail` already bounds scope and refuses unsupported task classes.
- `MPRetriever` now maps a typed nested `mp_filters` surface onto real `mpr.materials.summary.search(...)` kwargs and still has mock fallback behavior.
- `policy_filter` is now default-on for `band_gap_screening` as a single LLM-based post-retrieval chemistry screen.
- `DiscoveryPlan` now separates deterministic `source_universe` from parser-derived `requested_material_class`.
- `PropertyPredictor`, ranking, and reporting layers exist, but repo scope already exceeds narrow MVP needs.
- `StabilityChecker` uses MP `energy_above_hull` when available and otherwise returns honest `stability_unknown`.
- MLflow observability is wired in, but not required for basic MVP behavior.
- Offline benchmark tooling exists, and a provisional small baseline artifact set is now published for optional prediction/recalc regression tracking.
- Prompt-level QA now exists for 15 materials-informatics user questions covering supported screening, edge-case execution controls, and structured refusals through both `POST /discover` and `POST /discover/full`.
- Planner selective-recalc parsing now handles the natural phrasing `recalculate the top N candidates`.
- Property prediction now safely falls back when selective MatGL recalc excludes a candidate that also lacks an MP band gap.
- New `matsci` CLI now exists for presentable local/operator use:
  - `demo` renders compact shortlist output
  - `operator` renders full debug trace sections
  - `doctor` runs read-only setup diagnostics
  - `scenarios list|run` exposes built-in showcase requests
- CLI defaults to in-process execution and can target live HTTP API with `--api-url`.
- Policy filtering now runs by default after retrieval as one fail-closed LLM policy named `chemistry_screening`; no heuristic fallback path remains.
- Policy filter still applies only to `band_gap_screening`, but it now uses richer discovery context including parsed `mp_filters` and candidate duplicate metadata.
- Policy filter now frames candidates as Materials Project entries and consumes requested material class from user intent instead of old `material_class` shorthand.
- Live retrieval now re-applies core requested element/filter semantics client-side because upstream MP server filtering can be looser than user intent.
- Retrieval heuristics now avoid hard-coded material-class checks; class semantics stay above low-level retriever logic.
- Retrieval now dedupes exact `formula_pretty` and surfaces `has_multiple_entries` / `entry_count` in API and CLI responses.
- CLI `demo` and `operator` surfaces now display duplicate-entry metadata and report the new policy name.
- Parser material-class interpretation is now LLM-owned; deterministic regex class backfills were removed so code does not carry a hard-coded materials-class list.
- Policy screening now treats `research_goal` as authoritative, normalizes verbose LLM reasons, rejects class-mismatched molecular/salt-like high-gap entries, and runs a larger relaxed retrieval/screening pass when the first policy batch keeps zero candidates.
- Policy-screen candidate selection now prioritizes exact requested element-set matches before high band-gap sorting, so generic Si+C requests expose SiC-like candidates to the LLM screen instead of only high-gap complex formulas.
- Search-space expansion now runs before retrieval for supported screening tasks and fails closed if it cannot produce valid MP-compatible formula targets.
- `mp_property_screening` is now supported for generic MP-summary-queryable filters such as formation energy, hull energy, density, volume, formula, and chemsys.
- Retrieval now uses expansion formula targets first, then bounded chemsys fallback, instead of broad parser-only MP search for expanded requests.
- Generic MP-property results now skip MatGL prediction and return MP summary properties through compact API/CLI output.
- Search-space expansion now treats LLM-provided `chemsys` as advisory, computes canonical `chemsys` from formula, and retries when OpenRouter returns `{}` or omits `formula_targets`.
- MatGL integration now suppresses known third-party load-time warnings (`torchdata` deprecation banner and old-checkpoint `@model_version` banner) so `matsci demo --calculate-matgl` stays clean while retaining compatibility fallback behavior.

## Biggest Current Limitation
- Biggest current limitation is product-scope mismatch.
- Repo already contains broader screening-platform pieces than near-term MVP requires.
- Near-term work should reduce ambiguity around primary user flow before expanding scientific/platform depth.

## MVP Blockers
1. Simplify and validate `POST /discover` as primary user contract.
2. Confirm Materials Project retrieval quality for actual user query patterns.
3. Decide whether local prediction/recalc remains part of MVP or becomes secondary.
4. Keep chemistry filtering tightly bounded to supported MVP screening tasks.

## Non-MVP Future Work
1. Publish benchmark baselines if local prediction/recalc remains product-critical.
2. Broaden chemistry filtering beyond current `band_gap_screening` scope when there is a real execution path.
3. Expand planning into richer scientific ontology if product scope broadens.
4. Add richer external-facing reporting beyond raw `/discover/full` trace if compact output proves too thin.
5. Expand supported task classes only when real execution paths are added.
6. Decide whether CLI needs JSON/export modes, scenario files, or a richer TUI after operator/demo feedback.
7. Continue validating live MP retrieval quality for edge cases where real MP values exclude expected textbook materials.
8. Validate live expansion-target formula coverage against Materials Project for representative material-family queries.

## Bottom Line
Repository already contains a capable materials-screening scaffold, but near-term product target is narrower than current architecture.
MVP does not require benchmark publication, richer planning ontology, broader task coverage, or MLflow-heavy workflows.
Priority is clarifying and tightening retrieval-first user flow through `POST /discover`, while keeping advanced systems available as optional layers.
