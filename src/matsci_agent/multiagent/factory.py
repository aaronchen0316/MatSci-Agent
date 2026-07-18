from __future__ import annotations

from dataclasses import dataclass

from matsci_agent.multiagent.prompt_loader import load_agent_prompt
from matsci_agent.multiagent.schemas import (
    CodexDebuggerReport,
    FinalVerifierReport,
    MaterialsQueryCriticReport,
    RetrievalTesterReport,
)
from matsci_agent.multiagent.settings import MultiAgentSettings
from matsci_agent.multiagent.tools import ToolGroups


@dataclass(frozen=True)
class AgentRegistry:
    retrieval_tester: object
    materials_query_critic: object
    codex_debugger: object
    final_verifier: object


def build_agent_registry(sdk, settings: MultiAgentSettings, tool_groups: ToolGroups) -> AgentRegistry:
    """Create all agents in one place.

    Keeping agent construction centralized makes it easier to swap models,
    prompts, or tool surfaces without touching orchestration code.
    """

    retrieval_tester = sdk.Agent(
        name="Retrieval Tester Agent",
        instructions=load_agent_prompt("retrieval_tester"),
        model=settings.model,
        tools=tool_groups.tester,
        output_type=RetrievalTesterReport,
    )
    materials_query_critic = sdk.Agent(
        name="Materials Query Critic Agent",
        instructions=load_agent_prompt("materials_query_critic"),
        model=settings.model,
        tools=tool_groups.critic,
        output_type=MaterialsQueryCriticReport,
    )
    codex_debugger = sdk.Agent(
        name="Codex Debugger Agent",
        instructions=load_agent_prompt("codex_debugger"),
        model=settings.model,
        tools=tool_groups.debugger,
        output_type=CodexDebuggerReport,
    )
    final_verifier = sdk.Agent(
        name="Final Verifier Agent",
        instructions=load_agent_prompt("final_verifier"),
        model=settings.model,
        tools=tool_groups.verifier,
        output_type=FinalVerifierReport,
    )
    return AgentRegistry(
        retrieval_tester=retrieval_tester,
        materials_query_critic=materials_query_critic,
        codex_debugger=codex_debugger,
        final_verifier=final_verifier,
    )
