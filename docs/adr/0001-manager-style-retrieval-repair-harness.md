# ADR 0001: Retrieval Repair Harness With Python-Owned Retry Loop

## Status
Accepted

## Context
MatSci-Agent already has an internal deterministic retrieval workflow:
- chemistry/intent agent
- capability guardrail
- Search Space Expansion
- Materials Project retrieval
- Policy Filter
- ranking/reporting

What it does not have is an external evaluation and repair loop that can:
- grade retrieval quality
- diagnose stage failures
- open isolated code-fix branches
- commit bounded debugger changes
- re-run verification

This architecture is hard to reverse once prompts, tools, sessions, and review flow depend on it. Future readers would also ask why multi-agent code is outside `DiscoveryWorkflow` rather than inside it.

## Decision
Use a multi-agent retrieval-repair harness outside the existing workflow, but keep retry and stop-state sequencing in Python.

Implementation shape:
1. Keep agent specs/prompts under `agent_specs/`.
2. Keep runtime code under `src/matsci_agent/multiagent/`.
3. Use the OpenAI Agents SDK with one shared model/client configuration.
4. Default model: `gpt-5.4-mini`.
5. Make Python orchestrator own retry order and stop conditions: tester -> critic -> debugger -> verifier.
6. Do not create a Controller Agent. Python remains sole scheduler and emits final success only after Tester pass, Critic agreement, and real MP evidence.
7. Treat Final Verifier as a patch-acceptance gate. Its accepted patches require a fresh Tester + Critic cycle and cannot directly pass the harness.
8. Keep git mutation disabled by default behind an explicit env flag. Normal repair runs never push or open PRs.
9. Prefer isolated worktree branches for debugger changes, reuse one repair branch across retries, then remove clean worktrees while retaining branches.
10. Record non-secret run artifacts and typed agent handoffs under ignored local storage.
11. Preserve the current deterministic `DiscoveryWorkflow` as execution source of truth.
12. Rebind Tester and Critic tools to an accepted repair worktree; execute live evaluation in a subprocess importing that worktree's source tree.
13. Bind `repair-live` to one named live scenario and enforce its exact query, constraints, and quality assertions on initial and repaired-worktree evaluation.
14. Require deterministic repair-test evidence: changed test collection/pass, full-suite pass, no deleted or renamed tests, and no per-file line-coverage regression for changed production files.
15. Permit branch push and draft PR creation only through separate explicit `publish-pr`, with stored passing evidence, clean base/ancestry/diff checks, and `GITHUB_TOKEN`. Human review and GitHub Actions remain merge gates.

## Consequences
### Positive
- keeps repair loop separate from product retrieval path
- makes retry/state behavior deterministic and testable
- lets specialist prompts stay narrow
- supports OpenAI-compatible proxy endpoints through shared client configuration
- keeps live MP evals and git mutations opt-in
- preserves review evidence after temporary worktrees are removed
- avoids import confusion between repo prompt folder and `openai-agents` SDK

### Negative
- more orchestration logic now lives in Python and must stay covered by tests

### Follow-up
Add a separate reporting surface only if typed harness reports prove insufficient for human review. It must not own scheduling or stop conditions.
