# ADR 0001: Manager-Style Retrieval Repair Harness

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
Use a manager-style multi-agent harness outside the existing workflow.

Implementation shape:
1. Keep agent specs/prompts under `agent_specs/`.
2. Keep runtime code under `src/matsci_agent/multiagent/`.
3. Use the OpenAI Agents SDK with one shared model/client configuration.
4. Default model: `gpt-5.4-mini`.
5. Make the controller own the final answer and call specialists as bounded tools.
6. Keep git mutation and PR tools disabled by default behind explicit env flags.
7. Prefer isolated worktree branches for debugger changes.
8. Preserve the current deterministic `DiscoveryWorkflow` as execution source of truth.

## Consequences
### Positive
- keeps repair loop separate from product retrieval path
- gives controller one place to enforce guardrails
- lets specialist prompts stay narrow
- supports OpenAI-compatible proxy endpoints through shared client configuration
- keeps live MP evals and git mutations opt-in
- avoids import confusion between repo prompt folder and `openai-agents` SDK

### Negative
- manager-style orchestration is less deterministic than a fully code-owned loop

### Follow-up
If controller behavior becomes too free-form, move retry caps and stage routing into explicit code while keeping specialist agents and their schemas.
