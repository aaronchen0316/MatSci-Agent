# MatSci-Agent Context

## Purpose
MatSci-Agent is an agentic materials-screening system for bulk inorganic materials. The current product goal is natural-language-driven band-gap screening over Materials Project candidates, with optional local MatGL recalculation and bounded relaxation.

## Current System Shape
- Public entrypoint: `POST /discover`
- Orchestration: LangGraph workflow
- Retrieval source: Materials Project
- Primary target property: `band_gap`
- Property policy: prefer MP values, fall back to local MatGL models when data is missing or recalculation is explicitly requested
- Stability: currently mocked, not research-grade

## Core Terms

### Research Goal
The raw natural-language user request, such as:
"Find semiconductor materials with no silicon and band gap above 2 eV."

### Discovery Plan
A future typed planning object produced by the chemistry/intent agent. It should normalize the research goal into executable fields such as:
- `task_class`
- `parsed_constraints`
- `application_intent`
- `practicality_mode`
- `ranking_intent`
- `execution_policy`

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
The future interpretation agent responsible for converting the research goal into a typed `DiscoveryPlan`. This agent should reason about chemistry intent, application intent, ranking intent, and execution intent, but should not directly perform retrieval or prediction.

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
A future summarization agent that explains results or refusals after deterministic execution is complete. It should not modify candidate selection or scoring.

### Task Registry
A finite canonical registry of supported and unsupported task classes. This is the preferred guardrail mechanism because it is easier to audit, update, and test than free-form LLM capability judgments.

## Current Supported Scope
- bulk inorganic candidate retrieval from Materials Project
- band-gap screening
- optional bounded MatGL recalculation
- optional bounded structure relaxation
- mock stability filtering
- ranking and compact reporting

## Current Unsupported Scope
- diffusivity calculation
- long molecular dynamics trajectories
- transport-property simulation
- defect workflows
- arbitrary molecular simulation
- any workflow that exceeds bounded local compute assumptions

## Architectural Direction
The preferred next architecture is:
1. chemistry/intent agent
2. deterministic capability guardrail using a finite task registry
3. deterministic execution pipeline
4. reporting agent

This preserves agentic behavior where reasoning helps most, while keeping execution reproducible and debuggable.
