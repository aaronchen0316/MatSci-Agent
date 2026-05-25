# MatSci-Agent Context

## Purpose
MatSci-Agent is an agentic materials-screening system for bulk inorganic materials. The current product goal is retrieval-first natural-language-driven band-gap screening over Materials Project candidates, with optional local MatGL recalculation and bounded relaxation as secondary layers.

## Current System Shape
- Public entrypoint: `POST /discover`
- Debug trace entrypoint: `POST /discover/full`
- Orchestration: LangGraph workflow
- Retrieval source: Materials Project
- Primary target property: `band_gap`
- Property policy: prefer MP values, fall back to local MatGL models when data is missing or recalculation is explicitly requested
- Stability policy: use MP `energy_above_hull` when available, otherwise mark stability unknown
- Candidate selection policy: optional LLM-backed chemistry `policy_filter` after retrieval and before prediction, with fail-closed validation and one bounded replenish pass only when explicitly enabled
- Evaluation path: offline band-gap benchmark tooling against MP-known entries

## Core Terms

### Research Goal
The raw natural-language user request, such as:
"Find semiconductor materials with no silicon and band gap above 2 eV."

### Discovery Plan
A typed planning object produced by the chemistry/intent agent. It normalizes the research goal into executable fields such as:
- `task_class`
- `parsed_constraints`
- `application_intent`
- `practicality_mode`
- `ranking_intent`
- `execution_policy`

Current MVP emphasis:
- keep the plan thin and executable
- preserve broader schema compatibility
- treat richer non-MVP intent labels as secondary unless they drive live execution

### Task Class
A normalized label for the requested workflow. Examples:
- `band_gap_screening`
- `bulk_relaxation_only`
- `diffusivity_simulation`
- `molecular_dynamics`
- `transport_property_estimation`

Task class is the bridge between natural-language intent and the deterministic execution layer.

### Capability Assessment
A deterministic admission decision made after intent parsing. It determines whether the current codebase can execute the requested task within its current scientific scope and compute budget.

Expected fields:
- `supported`
- `reason_code`
- `reason_message`
- `closest_supported_mode`
- `suggested_next_action`

### Structured Refusal
A supported API outcome for requests outside current scope. The system should return a specific refusal instead of hallucinating execution for tasks like diffusivity or long MD.

### Chemistry / Intent Agent
Interpretation agent responsible for converting the research goal into a typed `DiscoveryPlan`. It reasons about chemistry intent, application intent, ranking intent, and execution intent, but does not directly perform retrieval or prediction.

### Deterministic Execution Pipeline
The current and future code path responsible for:
- retrieval
- policy filtering
- prediction
- optional relaxation
- stability
- ranking

This layer should remain typed, reproducible, and testable.

### Reporting Agent
Deterministic summarization layer that explains results or refusals after deterministic execution is complete. It should not modify candidate selection or scoring.

### Task Registry
A finite canonical registry of supported and unsupported task classes. This is the preferred guardrail mechanism because it is easier to audit, update, and test than free-form LLM capability judgments.

### Policy Filter
LLM-backed post-retrieval, pre-prediction chemistry filter driven by `DiscoveryPlan`.

Current behavior:
- only active for `band_gap_screening`
- disabled by default in retrieval-first MVP flow
- reuses the same provider/model configuration as the parser
- validates strict JSON decisions for every candidate in the batch
- fails closed on request, timeout, parse, or validation errors
- uses one bounded replenish pass when the first kept set underfills `top_k`

Current policy modes:
- `practical_screening`
- `exploratory_screening`

It records candidate-level fields such as:
- `filter_passed`
- `filter_reasons`
- `filter_policy`
- `filter_source`

### Stability Source
Origin of candidate stability evidence:
- `materials_project`
- `unknown`

This is carried on `StabilityResult` together with `method` and `used_relaxation`.

### Band-gap Benchmark
Offline evaluation path that samples MP entries with known `band_gap`, forces local predictor execution where needed, and reports:
- `mae`
- `rmse`
- `rank_correlation`
- failure/skipped counts
- MP passthrough vs MatGL vs fallback counts

Current published baseline:
- artifact set: `artifacts/bandgap_benchmark_small.{json,csv,md}`
- generated at: `2026-05-25T17:47:20.297358Z`
- sample size: `10`
- `mae=1.6451`
- `rmse=2.2116`
- `rank_correlation=0.7178`
- all 10 rows used MatGL path, with `fallback_count=0`

## Current Supported Scope
- bulk inorganic candidate retrieval from Materials Project
- band-gap screening
- optional bounded MatGL recalculation
- optional bounded structure relaxation
- MP-backed stability annotation with honest unknown state when MP hull data is missing
- optional LLM-backed practical vs exploratory chemistry filtering with fail-closed validation
- ranking and compact reporting
- offline benchmark artifact generation

## Current Limitations
- Stability is still weakest scientific layer:
  - MP `energy_above_hull` is authoritative when present.
  - when MP hull data is missing, current code returns stability unknown.
  - no local proxy or MatGL-based stability estimate is used.
- Chemistry filter only covers `band_gap_screening`.
- Chemistry filter reasons from compact metadata, not richer structure-aware features.
- Published benchmark baseline is intentionally small and only supports regression tracking for optional prediction/recalc behavior.
- Planning remains narrow and mostly parser-plus-regex enrichment.
- Reporting remains compact/deterministic rather than rich scientific analysis.

## Current Unsupported Scope
- diffusivity calculation
- long molecular dynamics trajectories
- transport-property simulation
- defect workflows
- arbitrary molecular simulation
- any workflow that exceeds bounded local compute assumptions

## Current Workflow
Current `DiscoveryWorkflow` shape:
1. chemistry/intent agent
2. deterministic capability guardrail using a finite task registry
3. Materials Project retrieval
4. optional LLM-backed `policy_filter`
5. deterministic prediction pipeline
6. stability evaluation
7. reporting agent

This preserves agentic behavior where reasoning helps most, while keeping execution reproducible and debuggable.

## Bottom Line
- Current repo is strong agentic engineering scaffold for bulk-inorganic band-gap screening.
- Main strengths:
  - typed contracts
  - deterministic capability admission
  - validated LLM chemistry filter
  - bounded expensive compute
  - reproducible local model packaging
- Main gap:
  - not yet research-grade materials discovery because chemistry scope is narrow and benchmark results are not yet published.

## Module Map

### API Entrypoint
- `src/matsci_agent/api/main.py`
- Caller-facing FastAPI shell for:
  - `POST /discover`
  - `POST /discover/full`
- `POST /discover` converts full workflow output into compact annotated `DiscoverySummaryResponse`
- `POST /discover/full` exposes full workflow trace for debugging, explainability, and interviews

### Workflow Skeleton
- `src/matsci_agent/workflow/graph.py`
- Central orchestrator for `DiscoveryWorkflow`
- Owns routing for:
  - capability refusal
  - optional chemistry-filter fail-closed stop
  - replenish pass
  - retrieval-first success path with stability as annotation/light ranking signal

### Workflow State
- `src/matsci_agent/workflow/state.py`
- Typed shared state carrying:
  - `DiscoveryPlan`
  - `CapabilityAssessment`
  - raw/filtered candidates
  - predictions
  - ranked candidates
  - provenance
  - status/messages

### Chemistry / Intent Agent
- `src/matsci_agent/agents/planner.py`
- Reuses parser output, then enriches deterministically into `DiscoveryPlan`
- Owns:
  - task classification
  - practicality mode
  - ranking intent
  - selective recalculation extraction

### Constraint Parser
- `src/matsci_agent/nlp/parser.py`
- Lowest-level LLM parsing layer
- Produces narrow `DiscoveryConstraints`
- Also provides JSON cleanup helper reused by chemistry filter

### Capability Guardrail
- `src/matsci_agent/guardrails/capability.py`
- Finite `Task Registry` implementation
- Maps `task_class` -> supported vs structured refusal

### Retriever
- `src/matsci_agent/tools/mp_retriever.py`
- Materials Project interface plus mock fallback
- Owns:
  - live MP query construction
  - client-side chemistry filtering
  - exclude-id replenish support
  - compact candidate feature payload

### Policy Filter
- `src/matsci_agent/tools/policy_filter.py`
- Optional LLM-backed chemistry filter for `band_gap_screening`
- Owns:
  - strict screening-judge prompt
  - provider/model reuse from parser config
  - fail-closed validation
  - candidate-level filter provenance

### Property Predictor
- `src/matsci_agent/tools/property_predictor.py`
- Deterministic property resolution layer
- Owns:
  - MP band-gap passthrough
  - local MEGNet prediction
  - optional TensorNet-based relaxation
  - selective `recalculate_top_n`

### Stability Checker
- `src/matsci_agent/tools/stability_checker.py`
- MP-first stability semantics
- Owns:
  - MP `energy_above_hull` path
  - honest unknown-stability path
  - stability-source provenance used as annotation and light ranking hint

### Reporting Agent
- `src/matsci_agent/agents/reporter.py`
- Deterministic reporting layer
- Summarizes:
  - refusal path
  - compact success path
  - caveats

### Evaluation
- `src/matsci_agent/evaluation/bandgap_benchmark.py`
- Offline benchmark library for predictor quality
- Used by:
  - `examples/benchmark_bandgap_predictor.py`

### Contracts
- `src/matsci_agent/schemas.py`
- Domain contract hub for:
  - `DiscoveryPlan`
  - `CapabilityAssessment`
  - `PolicyFilter*`
  - `StabilityResult`
  - benchmark artifacts
