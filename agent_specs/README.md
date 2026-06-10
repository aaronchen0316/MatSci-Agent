# Multi-Agent Retrieval Repair

This folder stores **agent specs and prompts**, not runtime Python packages.

Why:
- runtime code lives in `src/matsci_agent/multiagent/`
- prompt/spec files stay here in `agent_specs/`
- naming makes intent explicit, avoids confusion with SDK import name `agents`

## Architecture

Manager-style orchestration:
- Controller Agent
  - owns final decision
  - calls specialists in bounded order
- Retrieval Tester Agent
  - grades retrieval quality from offline traces first
  - may run live MP evals only when explicitly enabled
- Materials Query Critic Agent
  - maps failure -> root cause -> owning module
- Codex Debugger Agent
  - patches code in isolated worktree branch
  - can commit and open PR only when env gates allow it
- Final Verifier Agent
  - reviews debugger output
  - decides pass / fail / needs tester update

## Key design rules

1. Keep repair loop outside `DiscoveryWorkflow`.
2. Keep retrieval execution deterministic inside app.
3. Give agents narrow tools, not generic unrestricted shell.
4. Default to offline evals and read-only git behavior.
5. Use one shared model client for all sub-agents.

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
- no PR creation

Enable only when ready:
- `MULTIAGENT_ENABLE_LIVE_MP=1`
- `MULTIAGENT_ENABLE_GIT_WRITE=1`
- `MULTIAGENT_ENABLE_PRS=1`

## Install

```bash
uv sync --extra dev --extra agents
```

## Entry point

```bash
uv run matsci-multiagent plan "Eval and repair retrieval quality for current code base"
```

Later, when ready to run real model calls:

```bash
export MULTIAGENT_API_KEY="..."
export MULTIAGENT_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MULTIAGENT_MODEL="gpt-5.4-mini"
uv run matsci-multiagent run "Eval and repair retrieval quality for current code base"
```
