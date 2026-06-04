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
- `policy_filter` stays in MVP as default-on chemistry-quality guard after retrieval.
- `stability_checker` stays only as MP-backed annotation, not as scientific expansion target.
- `POST /discover/full` stays as debug/demo surface, not primary user contract.
- MLflow stays optional infra, not MVP requirement.
- Benchmark publication, richer planning ontology, broader unsupported-task taxonomy, and richer external reporting are deferred unless prediction-heavy workflow remains core.
- Reason: current repo already exceeds desired MVP scope; priority is reducing product ambiguity, not adding more platform depth.

## Decision 23: Add prompt-level QA coverage with a test-only science-oriented retriever fixture
- Add automated QA for 15 materials-informatics prompts across supported screening, execution-control edge cases, and structured refusals.
- Use a deterministic parser double and richer offline retriever fixture in tests so prompt QA validates workflow semantics without requiring live MP or LLM access.
- Reason: this gives stable regression coverage for the user-facing control plane while keeping production behavior unchanged.

## Decision 24: Selective MatGL recalc should degrade safely for out-of-window candidates
- When `recalculate_top_n` is set, candidates outside the selected recalc window that also lack an MP band gap must fall back to heuristic prediction instead of crashing.
- Reason: selective recalc narrows expensive compute; it must not assume every unselected candidate still has an MP value available.

## Decision 25: Add hybrid local CLI with separate demo and operator surfaces
- Add `matsci` console entrypoint with commands:
  - `demo` for polished compact shortlist rendering
  - `operator` for full trace/debug rendering
  - `doctor` for read-only environment diagnostics
  - `scenarios list|run` for built-in demo presets
- CLI defaults to in-process workflow execution and accepts optional `--api-url` to hit live `/discover` and `/discover/full`.
- Reason: gives presentable operator/demo surface without changing core API or workflow contracts.

## Decision 26: Make policy filtering default-on
- Post-retrieval `policy_filter` is now enabled by default in workflow execution instead of being gated by `MATSCI_ENABLE_POLICY_FILTER`.
- Reason: user-facing runs should always have an intent-aware post-retrieval filter.

## Decision 27: Collapse policy filtering to one fail-closed LLM chemistry screen
- Remove `practical_screening` / `exploratory_screening` runtime modes and replace them with one policy name: `chemistry_screening`.
- Keep only deterministic hard rejects for impractical or radioactive elements as explicit guardrail.
- Main screening decision is LLM-first and must return one decision per candidate; invalid, incomplete, or unavailable LLM responses stop execution.
- Reason: chemistry-fit judgment is better handled by one constrained materials-science screen than by brittle branching heuristics.

## Decision 28: Expand retrieval around real MP search kwargs with typed nested filters
- `DiscoveryConstraints` now carries nested typed `mp_filters` matching a supported subset of real `mpr.materials.summary.search(...)` kwargs.
- Legacy aliases remain public for compatibility and are mapped into `mp_filters` when present.
- Retrieval omits undefined filters instead of inventing defaults in the public request contract.
- Reason: aligns parsing and retrieval with actual Materials Project capabilities without breaking existing callers.

## Decision 29: Dedupe retrieval by exact `formula_pretty` and surface duplicate metadata
- Retrieval now collapses repeated exact formulas to a single highest-ranked representative.
- Candidate responses expose `has_multiple_entries` and `entry_count`, and the same metadata is mirrored into `features` for raw traces.
- Reason: v1 shortlist should not repeat the same chemistry while still signaling that multiple MP entries exist.

## Decision 30: Split source truth from user-requested material class
- Replace narrow `material_class` field with:
  - `source_universe` for deterministic backend/source truth
  - `requested_material_class` for parser-derived user-intent class
- `source_universe` is currently `materials_project_entries` for this workflow.
- `requested_material_class` is normalized to snake_case and falls back to `unknown`.
- Reason: downstream policy reasoning should know candidate source without hard-coding a pre-baked material ontology onto the user request.

## Decision 31: Policy filter should frame candidates as Materials Project entries
- Policy-filter prompt no longer tells LLM candidates are `bulk_inorganic` or any other preexisting material class.
- Prompt now says candidates are Materials Project entries and instructs model to use `requested_material_class` when provided.
- Reason: Materials Project source universe and requested material class are different concepts and must not be collapsed.

## Decision 32: Enforce client-side subset filtering after live MP retrieval
- Live Materials Project `summary.search(...)` results are not trusted to satisfy all requested element/filter semantics exactly on their own.
- Retriever now re-applies core element, boolean, and numeric filters client-side before policy filtering.
- Reason: user queries like `contains Si and C` were returning actinide entries from upstream server-side search behavior; local enforcement is required for believable shortlist quality.

## Decision 33: Do not hard-code requested material classes inside retrieval heuristics
- Retriever/ranker heuristics must not branch on specific requested material classes such as `semiconductor`.
- Material-class semantics belong in parsed intent, explicit filters, policy-screening payload, and future generic class-aware reasoning layers.
- Reason: class-specific keyword heuristics leak domain assumptions into low-level retrieval behavior and make the system brittle outside one materials class.

## Decision 34: Let LLM own material-class intent and use generic candidate exposure
- Remove deterministic requested-material-class regex backfills from parser/planner.
- Keep deterministic parsing only for generic controls such as elements, band-gap bounds, result count, and recalc intent.
- Policy filter uses `research_goal` as authoritative and `requested_material_class` only as a parser hint.
- When the first policy batch keeps zero candidates, retrieve a larger batch and screen more candidates before stopping.
- Candidate selection for policy screening prioritizes exact requested element-set matches and simpler element sets before band-gap sorting.
- Reason: queries such as `semiconductor materials with silicon and carbon` were showing high-gap molecular/salt-like MP entries because band-gap-first selection hid better semantic candidates from the LLM screen.

## Assumptions
- MVP means retrieval-first product, not research-grade predictor platform.
- Existing files only; no new `memory/project.md`, no new log file.
- If later decision is "prediction/recalc remains core differentiator," benchmark publication returns to high priority.
