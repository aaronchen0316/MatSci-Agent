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
- commit or propose PRs
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
6. Keep controller prompt only as optional summarizer, not sequencing hot path.
7. Keep git mutation and PR tools disabled by default behind explicit env flags.
8. Prefer isolated worktree branches for debugger changes and reuse one repair branch across retries.
9. Preserve the current deterministic `DiscoveryWorkflow` as execution source of truth.

## Consequences
### Positive
- keeps repair loop separate from product retrieval path
- makes retry/state behavior deterministic and testable
- lets specialist prompts stay narrow
- supports OpenAI-compatible proxy endpoints through shared client configuration
- keeps live MP evals and git mutations opt-in
- avoids import confusion between repo prompt folder and `openai-agents` SDK

### Negative
- more orchestration logic now lives in Python and must stay covered by tests

### Follow-up
If a controller summary layer becomes useful later, keep it outside the retry hot path and feed it typed harness reports instead of letting it own sequencing.
