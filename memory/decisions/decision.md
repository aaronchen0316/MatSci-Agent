# MatSci-Agent Key Decisions

## Decision 1: Band-gap-first objective
- Switched objective to semiconductor band-gap screening.
- Reason: available pretrained MatGL/MEGNet models and direct compatibility with MP `band_gap` field.

## Decision 2: Hybrid property strategy
- MP band gap is authoritative when present.
- MatGL is used for missing values or forced recalculation.
- Reason: minimizes unnecessary compute and preserves source fidelity.

## Decision 3: Guardrails for forced MatGL recalculation
- Enforce recalc cap (`10`) and atom-count cap (`< 50`).
- Reason: keep local runtime practical and predictable.

## Decision 4: Python/runtime pinning for DGL-backed model
- Pin project runtime to Python 3.12 and DGL-compatible torch stack.
- Reason: `MEGNet-MP-2019.4.1-BandGap-mfi` depends on DGL path not reliable on Python 3.13.

## Decision 5: Structured fallback chain
- MatGL primary -> compatibility fallback -> heuristic fallback.
- Reason: maintain service availability even when model/runtime mismatch occurs.

## Decision 6: Install MatGL from source instead of relying on older PyPI behavior
- Replace the PyPI `matgl` package in the project venv with an editable install from the upstream repository.
- Reason: the older installed package did not support the documented Hugging Face repo-id model loading path correctly.

## Decision 7: Use local downloaded model bundles by default
- Default band-gap path: `models/pretrained/MEGNet-MP-2019.4.1-BandGap-mfi`
- Default relaxation path: `models/pretrained/TensorNet-PES-MatPES-PBE-2025.2`
- Reason: local bundles are more reproducible and avoid dependence on live remote model resolution at runtime.

## Decision 8: Prefer goal-conditioned filtering over semiconductor-only hard-coded filtering
- Do not bake a single explicit semiconductor practicality filter into the pipeline.
- Reason: system should generalize to different chemistry tasks and application intents.

## Decision 9: Multi-agent direction should augment, not replace, deterministic orchestration
- Use agents for chemistry intent understanding and result summarization, but keep deterministic schemas and execution boundaries for core toolchain.
- Reason: preserves reproducibility, debuggability, and provenance while allowing richer agentic behavior.

## Decision 10: Add explicit capability guardrail before execution
- Introduce deterministic capability-assessment stage after intent parsing and before retrieval/prediction.
- Reason: system must refuse unsupported requests like diffusivity or long MD rather than hallucinating execution.

## Decision 11: Use finite task registry for capability admission
- Capability admission based on finite canonical task registry, not free-form LLM judgments of what system can do.
- Reason: registry easier to audit, update, and test, and avoids over-admitting unsupported workflows.

## Decision 12: Preferred multi-agent architecture
- Target architecture:
  1. chemistry/intent agent,
  2. deterministic capability guardrail backed by task registry,
  3. chemistry-aware filtering,
  4. deterministic execution pipeline,
  5. reporting agent.
- Reason: gives agentic reasoning where it adds leverage while keeping tool execution reproducible and bounded.

## Decision 13: Implement planning and refusal without unnecessary model calls
- Planning agent reuses existing parser call and enriches result deterministically.
- Reporting layer defaults to deterministic summarization.
- Reason: keeps architecture agentic while avoiding unnecessary API cost and failure modes.

## Decision 14: Represent selective recalculation explicitly in execution policy
- Introduce `recalculate_top_n` in execution policy and property predictor input.
- Reason: prompts like `redo top 1 candidate` should not degrade to global forced recalculation.

## Decision 15: Honest stability simplification
- Use MP `energy_above_hull` when available; otherwise return `stability_unknown`.
- Reason: better to expose missing stability honestly than imply scientific confidence from fake proxy values.

## Decision 16: Replace active deterministic policy filter with LLM chemistry filter
- Active filter path is now LLM-backed for `band_gap_screening` tasks only.
- Reason: chemistry plausibility needs more domain knowledge than brittle hand-coded heuristics.

## Decision 17: Chemistry filter must fail closed
- Request, timeout, parse, or validation failures stop execution before prediction and surface a human-readable API failure.
- Reason: better to refuse than silently pass bad chemistry decisions downstream.

## Decision 18: Chemistry filter shares parser provider/model config
- No separate filter provider env vars were introduced.
- Reason: simpler runtime configuration and fewer moving parts.

## Decision 19: Chemistry filter uses one bounded replenish pass
- If first kept set underfills `top_k`, fetch unseen candidates once and run one more filter pass.
- Reason: improves result fill rate without turning filtering into an open-ended loop.

## Decision 20: Benchmarking should be offline script + reusable library
- Benchmark logic lives in reusable evaluation module plus runnable example script.
- Reason: easier to test, easier to rerun locally, no need to expose through API.

## Decision 21: Add `/discover/full` as debug-trace endpoint
- Keep `POST /discover` compact and add separate `POST /discover/full` that exposes raw workflow artifacts.
- Reason: improves debugging, explainability, and interview demos without complicating core execution flow.

## Decision 22: Re-scope near-term product to MVP retrieval-first workflow
- Near-term product goal is LLM-based user query parsing, Materials Project retrieval, and compact output through `POST /discover`.
- `policy_filter` stays in MVP as lightweight chemistry-quality guard when useful.
- `policy_filter` is disabled by default and only runs when explicitly enabled through runtime configuration.
- `stability_checker` stays only as MP-backed annotation, not as scientific expansion target.
- `POST /discover/full` stays as debug/demo surface, not primary user contract.
- MLflow stays optional infra, not MVP requirement.
- Benchmark publication, richer planning ontology, broader unsupported-task taxonomy, and richer external reporting are deferred unless prediction-heavy workflow remains core.
- Reason: current repo already exceeds desired MVP scope; priority is reducing product ambiguity, not adding more platform depth.

## Assumptions
- MVP means retrieval-first product, not research-grade predictor platform.
- Existing files only; no new `memory/project.md`, no new log file.
- If later decision is "prediction/recalc remains core differentiator," benchmark publication returns to high priority.
