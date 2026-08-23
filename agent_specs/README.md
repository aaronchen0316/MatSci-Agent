# Multi-Agent Retrieval Repair

This folder stores **agent specs and prompts**, not runtime Python packages.

Why:
- runtime tooling lives in `src/multiagent/`, outside product package `src/matsci_agent/`
- prompt/spec files stay here in `agent_specs/`
- naming makes intent explicit, avoids confusion with SDK import name `agents`

## Architecture

Python-owned orchestration:
- Python schedules specialists in fixed order and owns retry/stop conditions.
- Retrieval Tester Agent
  - grades retrieval quality from offline traces first
  - may run live MP evals only when explicitly enabled
- Materials Query Critic Agent
  - independently approves or disputes Tester conclusions from immutable evidence
- Codex Debugger Agent
  - patches code in isolated worktree branch
  - can commit only when mutation mode is enabled
- Final Verifier Agent
  - accepts or rejects debugger patch quality
  - never declares final harness success

## Key design rules

1. Keep repair loop outside `DiscoveryWorkflow`.
2. Keep retrieval execution deterministic inside app.
3. Give agents narrow tools, not generic unrestricted shell.
4. Default to offline evals and read-only git behavior.
5. Use one shared model client for all sub-agents.
6. Python emits `dual_review_pass` only after Tester pass, Critic agreement, and real MP evidence.
7. Final Verifier is a patch-acceptance gate; no Controller Agent exists.

## API key / proxy answer

You do **not** need one key per sub-agent.

Proper setup:
- one shared client at harness startup
- all sub-agents reuse that client

OpenAI Agents SDK supports OpenAI-compatible endpoints with custom `base_url` and `api_key`.

That means:
- real OpenAI key works
- proxy key can work **if proxy is truly OpenAI-compatible**

Recommended env:
- `MULTIAGENT_API_KEY`
- `MULTIAGENT_BASE_URL`
- `MULTIAGENT_MODEL=gpt-5.4-mini`
- `MULTIAGENT_MAX_TURNS=30`

Tracing note:
- if you use non-OpenAI proxy key, disable tracing by default
- or provide separate real OpenAI tracing key later

Official docs:
- Agents SDK config supports custom `AsyncOpenAI(base_url=..., api_key=...)`
- OpenAI docs recommend disabling tracing when you do not have a platform OpenAI key

## Safety gates

Default:
- no live MP evals
- no git writes
- no branch push or PR creation

Enable only when ready:
- `MULTIAGENT_ENABLE_LIVE_MP=1`
- `MULTIAGENT_ENABLE_GIT_WRITE=1`
- `MULTIAGENT_ARTIFACT_ROOT=artifacts/multiagent-runs`
- `MULTIAGENT_REPAIR_BRANCH_PREFIX=fix`
- `MULTIAGENT_TARGET_BASE_BRANCH=main`
- optional `MULTIAGENT_TARGET_REPO=/path/to/product-repository`

## Install

```bash
uv sync --extra dev --extra agents
```

## Entry point

```bash
MULTIAGENT_ENABLE_LIVE_MP=1 uv run matsci-multiagent validate
MULTIAGENT_ENABLE_LIVE_MP=1 MULTIAGENT_ENABLE_GIT_WRITE=1 uv run matsci-multiagent validate-repair
MULTIAGENT_ENABLE_LIVE_MP=1 MULTIAGENT_ENABLE_GIT_WRITE=1 uv run matsci-multiagent validate-repair --adopt fix/example
```

Both commands preflight `gpt-5.4-mini` through the configured proxy and need MP plus LLM credentials. They write evidence under the ignored artifact root.

All repairs start from current `origin/main`; no local `main` checkout is modified. Explicit `--adopt` branches require matching stored Debugger evidence, receive fresh proof, and may get one same-branch retry. `validate-repair` publishes only fully proven `fix/<issue>` results with exact-SHA CI evidence. SSH pushes `fix/<issue>`, GitHub Actions validates the exact PR SHA, then the harness requests a squash merge to `main`. `GITHUB_TOKEN` is only used for GitHub API calls; no token is printed or stored in artifacts.
