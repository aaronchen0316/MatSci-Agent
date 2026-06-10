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
    disable_tracing: bool = True
    max_controller_turns: int = 30
    enable_live_mp: bool = False
    enable_git_write: bool = False
    enable_prs: bool = False
    base_branch: str = "multi-agent"
    github_repo: str | None = None
    worktree_root: Path = Path("/tmp/matsci-agent-worktrees")

    @classmethod
    def from_env(cls, repo_root: str | Path | None = None) -> "MultiAgentSettings":
        root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
        model = os.getenv("MULTIAGENT_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
        return cls(
            repo_root=root,
            model=model,
            api_key=(os.getenv("MULTIAGENT_API_KEY") or os.getenv("OPENAI_API_KEY")),
            base_url=(os.getenv("MULTIAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
            disable_tracing=os.getenv("MULTIAGENT_DISABLE_TRACING", "1").lower() not in {"0", "false", "no"},
            max_controller_turns=int(os.getenv("MULTIAGENT_MAX_CONTROLLER_TURNS", "30")),
            enable_live_mp=os.getenv("MULTIAGENT_ENABLE_LIVE_MP", "0") in {"1", "true", "yes"},
            enable_git_write=os.getenv("MULTIAGENT_ENABLE_GIT_WRITE", "0") in {"1", "true", "yes"},
            enable_prs=os.getenv("MULTIAGENT_ENABLE_PRS", "0") in {"1", "true", "yes"},
            base_branch=os.getenv("MULTIAGENT_BASE_BRANCH", "multi-agent"),
            github_repo=os.getenv("MULTIAGENT_GITHUB_REPO"),
            worktree_root=Path(os.getenv("MULTIAGENT_WORKTREE_ROOT", "/tmp/matsci-agent-worktrees")),
        )
