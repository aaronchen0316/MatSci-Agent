# Multi-Agent Harness Report

## Purpose
The multi-agent harness evaluates and, when explicitly enabled, repairs retrieval quality outside `DiscoveryWorkflow`. The deterministic product workflow remains the source of truth for discovery results.

Python owns scheduling, retry limits, and terminal state. There is no Controller Agent.

## Specialists
| Specialist | Responsibility | Cannot do |
| --- | --- | --- |
| Retrieval Tester | Runs retrieval evaluation and reports pass, failure stage, or block. | Modify code. |
| Materials Query Critic | Independently reviews the Tester conclusion using immutable typed evidence and chemistry/materials reasoning. | Make a second live MP query. |
| Codex Debugger | Diagnoses and patches approved repair targets in an isolated worktree. | Push or open a PR. |
| Final Verifier | Accepts, rejects, blocks, or requests retest for a patch. | Declare final harness success. |

## Evidence Rules
Tester attaches typed live-evaluation evidence. It includes compiled filters, provenance, constraint violations, counts, and bounded candidate snapshots:

- first 20 raw candidates
- first 20 filtered candidates
- all ranked candidates

Each snapshot contains MP ID, formula, elements, relevant MP properties, symmetry, policy decision/reasons, and ranking/stability fields. It excludes structures and arbitrary feature payloads.

`dual_review_pass` requires all of:

1. Tester status is `pass`.
2. Critic verdict is `agree`.
3. Live evaluation status is `pass`.
4. Provenance confirms real Materials Project use.
5. Raw, filtered, and ranked snapshots are present.

Offline-only, mock-fallback, unavailable, or incomplete evidence cannot produce a pass.

## Decision Rules
| Tester | Critic | Action |
| --- | --- | --- |
| `blocked` | not called | Terminal `tester_blocked`. |
| `pass` | `agree` | Python checks evidence. Valid evidence -> `dual_review_pass`; otherwise `scientific_evidence_blocked`. |
| `pass` | `disagree` | Send Tester/Critic evidence to Debugger, then Final Verifier. |
| `fail` | `agree` | Send Tester/Critic evidence to Debugger, then Final Verifier. |
| `fail` | `disagree` | Refresh Tester before any patch, with Critic feedback. |
| any non-blocked result | `blocked` | Terminal `critic_blocked`. |

Critic uses `agree`, `disagree`, or `blocked`:

- `agree`: no material findings; informational notes only.
- `disagree`: at least one material finding; may include owning modules and repair guidance.
- `blocked`: insufficient evidence for independent review.

## Repair and Retest
Debugger mutations require explicit Git-write opt-in and stay in an isolated worktree. Final Verifier outcomes:

- `accepted`: patch is acceptable, but must receive a fresh Tester + Critic cycle.
- `needs_tester_refresh`: fresh Tester + Critic cycle required.
- `fail` or `blocked`: terminal outcome.

Tester refresh feedback is typed and records whether it came from Critic or Final Verifier. After a Debugger patch receives `accepted` or `needs_tester_refresh`, the next Tester and Critic cycle is rebound to the repair worktree. Live evaluation runs in a child Python process with that worktree's `src/` first on `PYTHONPATH`, so evidence covers patched code rather than the parent checkout.

The harness allows three total Tester/Critic review cycles. A requested refresh after cycle three ends with `review_cycle_exhausted`; an accepted patch never bypasses this limit.

The harness runs from the tooling-only `multi-agent` branch, but evaluates a detached worktree from product `main`. `repair-live --scenario <name>` binds one live regression contract, creates only `fix/<issue>` worktrees from `main`, and injects the exact query and constraints into every Tester cycle. Scoped evaluation imports product `src/` before tooling `src/`, so evidence always covers `main` or the repaired product worktree.

Production live repairs require deterministic test proof before Verifier acceptance can become useful:

1. A committed repair changes production Python and one or more Python tests.
2. Debugger reports exact changed test files and targets.
3. Changed tests collect and pass, then the full suite passes.
4. Per-file line coverage for every changed production file does not decrease from the base branch.
5. Deleted, renamed, malformed, duplicate, unrelated, or coverage-reducing test edits are rejection evidence for Final Verifier.

## Artifacts and Safety
Each run writes non-secret typed inputs, reports, evaluator evidence, patch/commit evidence, and cleanup results under ignored `artifacts/multiagent-runs/<run-id>/`.

Default mode is read-only and offline. Live MP evaluation and Git writes are independent opt-ins. Normal `run` never pushes, opens PRs, or merges; only fully passing `repair-live` can publish a product repair.

After a repaired live scenario passes every gate, `repair-live` pushes its product-only `fix/<issue>` branch through SSH, creates a ready PR to `main`, waits for GitHub Actions on that exact head SHA, then requests a squash merge through the GitHub API. Any failed or timed-out gate retains the branch and artifacts without merging. `GITHUB_TOKEN` is used only for GitHub API calls and needs Contents plus Pull requests read/write permission; SSH remains the push transport.

Worktree tools require a registered Git worktree below the configured worktree root. File mutations reject traversal even when an input begins with an allowlisted prefix; structured pytest targets must resolve under `tests/`.
