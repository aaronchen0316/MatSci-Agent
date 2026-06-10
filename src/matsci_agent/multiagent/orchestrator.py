from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from matsci_agent.multiagent.factory import AgentRegistry, build_agent_registry
from matsci_agent.multiagent.schemas import (
    CodexDebuggerInput,
    FinalVerifierInput,
    MaterialsQueryCriticInput,
    RetrievalTesterInput,
)
from matsci_agent.multiagent.sdk import configure_sdk
from matsci_agent.multiagent.settings import MultiAgentSettings
from matsci_agent.multiagent.tools import build_tool_groups


@dataclass
class MultiAgentHarness:
    settings: MultiAgentSettings
    sdk: Any
    registry: AgentRegistry

    @classmethod
    def build(cls, settings: MultiAgentSettings | None = None) -> "MultiAgentHarness":
        runtime = settings or MultiAgentSettings.from_env()
        sdk = configure_sdk(runtime)
        tool_groups = build_tool_groups(sdk, runtime)

        # We wrap sub-agents as function tools here instead of using Agent.as_tool()
        # directly. This gives us explicit control over:
        # - input schemas
        # - typed final-output serialization
        # - future per-agent logging / branch policy hooks
        temp_registry = build_agent_registry(sdk, runtime, tool_groups, controller_tools=[])
        runner = sdk.Runner

        async def run_retrieval_tester(payload: RetrievalTesterInput) -> dict[str, Any]:
            result = await runner.run(
                temp_registry.retrieval_tester,
                payload.model_dump_json(indent=2),
            )
            return result.final_output.model_dump(mode="json")

        async def run_materials_query_critic(payload: MaterialsQueryCriticInput) -> dict[str, Any]:
            result = await runner.run(
                temp_registry.materials_query_critic,
                payload.model_dump_json(indent=2),
            )
            return result.final_output.model_dump(mode="json")

        async def run_codex_debugger(payload: CodexDebuggerInput) -> dict[str, Any]:
            result = await runner.run(
                temp_registry.codex_debugger,
                payload.model_dump_json(indent=2),
            )
            return result.final_output.model_dump(mode="json")

        async def run_final_verifier(payload: FinalVerifierInput) -> dict[str, Any]:
            result = await runner.run(
                temp_registry.final_verifier,
                payload.model_dump_json(indent=2),
            )
            return result.final_output.model_dump(mode="json")

        controller_tools = [
            sdk.function_tool(run_retrieval_tester),
            sdk.function_tool(run_materials_query_critic),
            sdk.function_tool(run_codex_debugger),
            sdk.function_tool(run_final_verifier),
            *tool_groups.shared,
        ]
        registry = build_agent_registry(sdk, runtime, tool_groups, controller_tools=controller_tools)
        return cls(settings=runtime, sdk=sdk, registry=registry)

    async def run(self, objective: str) -> dict[str, Any]:
        """Run controller once.

        Controller owns internal specialist order through its tools. Keep this
        thin. If you later need strict retry caps or durable resume behavior,
        move loop control into Python around the controller.
        """

        system_context = {
            "objective": objective,
            "repo_root": str(self.settings.repo_root),
            "model": self.settings.model,
            "allow_live_mp": self.settings.enable_live_mp,
            "allow_git_write": self.settings.enable_git_write,
            "allow_prs": self.settings.enable_prs,
            "base_branch": self.settings.base_branch,
            "github_repo": self.settings.github_repo,
        }
        prompt = (
            "Run retrieval-repair orchestration for this objective.\n"
            "Use specialist tools in required order from controller prompt.\n"
            "Return final ControllerSummary only.\n\n"
            f"{json.dumps(system_context, indent=2)}"
        )
        result = await self.sdk.Runner.run(
            self.registry.controller,
            prompt,
            max_turns=self.settings.max_controller_turns,
        )
        return result.final_output.model_dump(mode="json")
