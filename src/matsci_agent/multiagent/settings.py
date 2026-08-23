from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MultiAgentSettings:
    """Shared runtime config for every sub-agent in the harness.

    One shared client/config is the proper default. Do not create one API key
    per sub-agent unless you intentionally need separate billing or routing.
    """

    repo_root: Path
    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    base_url: str | None = None
    max_agent_turns: int = 20
    disable_tracing: bool = True
    enable_live_mp: bool = False
    enable_git_write: bool = False
    base_branch: str = "multi-agent"
    repair_branch_prefix: str = "retrieval-fix"
    worktree_root: Path = Path("/tmp/matsci-agent-worktrees")
    artifact_root: Path | None = None

    @property
    def resolved_artifact_root(self) -> Path:
        return (self.artifact_root or self.repo_root / "artifacts" / "multiagent-runs").resolve()

    @classmethod
    def from_env(cls, repo_root: str | Path | None = None) -> "MultiAgentSettings":
        root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
        model = os.getenv("MULTIAGENT_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
        max_agent_turns = max(1, int(os.getenv("MULTIAGENT_MAX_TURNS", "20")))
        artifact_root = Path(os.getenv("MULTIAGENT_ARTIFACT_ROOT", "artifacts/multiagent-runs")).expanduser()
        if not artifact_root.is_absolute():
            artifact_root = root / artifact_root
        return cls(
            repo_root=root,
            model=model,
            api_key=(os.getenv("MULTIAGENT_API_KEY") or os.getenv("OPENAI_API_KEY")),
            base_url=(os.getenv("MULTIAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
            max_agent_turns=max_agent_turns,
            disable_tracing=os.getenv("MULTIAGENT_DISABLE_TRACING", "1").lower() not in {"0", "false", "no"},
            enable_live_mp=os.getenv("MULTIAGENT_ENABLE_LIVE_MP", "0") in {"1", "true", "yes"},
            enable_git_write=os.getenv("MULTIAGENT_ENABLE_GIT_WRITE", "0") in {"1", "true", "yes"},
            base_branch=os.getenv("MULTIAGENT_BASE_BRANCH", "multi-agent"),
            repair_branch_prefix=os.getenv("MULTIAGENT_REPAIR_BRANCH_PREFIX", "retrieval-fix"),
            worktree_root=Path(os.getenv("MULTIAGENT_WORKTREE_ROOT", "/tmp/matsci-agent-worktrees")),
            artifact_root=artifact_root,
        )
