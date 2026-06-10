from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from matsci_agent.multiagent.settings import MultiAgentSettings

_READONLY_PREFIXES = {
    ("pwd",),
    ("ls",),
    ("find",),
    ("sed",),
    ("rg",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "branch"),
    ("git", "log"),
}


@dataclass(frozen=True)
class ToolGroups:
    shared: list[object]
    tester: list[object]
    critic: list[object]
    debugger: list[object]
    verifier: list[object]


def build_tool_groups(sdk, settings: MultiAgentSettings) -> ToolGroups:
    """Create narrow tool surfaces per specialist.

    Proper multi-agent setup gives each specialist only tools it needs. Avoid a
    single unrestricted shell tool for every agent.
    """

    def _repo_file(relative_path: str) -> Path:
        path = (settings.repo_root / relative_path).resolve()
        if settings.repo_root not in path.parents and path != settings.repo_root:
            raise ValueError(f"path escapes repo root: {relative_path}")
        return path

    def _run(args: list[str], cwd: Path | None = None) -> str:
        result = subprocess.run(
            args,
            cwd=str(cwd or settings.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()

    def _readonly_allowed(argv: list[str]) -> bool:
        for prefix in _READONLY_PREFIXES:
            if tuple(argv[: len(prefix)]) == prefix:
                return True
        return False

    def read_context_snapshot() -> str:
        """Return compact context needed by retrieval-repair agents."""

        context_path = settings.repo_root / "CONTEXT.md"
        readme_path = settings.repo_root / "README.md"
        parts = [
            "# CONTEXT.md\n",
            context_path.read_text()[:12000],
            "\n\n# README.md\n",
            readme_path.read_text()[:12000],
        ]
        return "".join(parts)

    def read_repo_file(relative_path: str, start_line: int = 1, end_line: int = 240) -> str:
        """Read repo file slice. Good for focused review without full repo dump."""

        path = _repo_file(relative_path)
        lines = path.read_text().splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        selected = lines[start - 1 : end]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start)]
        return "\n".join(numbered)

    def list_repo_files(pattern: str = "src") -> str:
        """List files. Use before asking for individual file slices."""

        args = ["find", pattern, "-type", "f"] if pattern != "." else ["find", ".", "-type", "f"]
        return _run(args)

    def run_readonly_repo_command(command: str) -> str:
        """Run narrow read-only commands only.

        This is safer than exposing generic shell to all agents.
        """

        argv = shlex.split(command)
        if not argv or not _readonly_allowed(argv):
            raise ValueError(f"command not allowed in read-only tool: {command}")
        return _run(argv)

    def create_branch_worktree(branch_name: str) -> str:
        """Create isolated worktree for debugger.

        Default off. Enable only when you trust harness behavior.
        """

        if not settings.enable_git_write:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "git writes disabled",
                }
            )
        settings.worktree_root.mkdir(parents=True, exist_ok=True)
        worktree_path = settings.worktree_root / branch_name
        output = _run(
            [
                "git",
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_path),
                settings.base_branch,
            ]
        )
        return json.dumps({"status": "created", "branch_name": branch_name, "worktree_path": str(worktree_path), "output": output})

    def read_worktree_diff(worktree_path: str) -> str:
        """Read diff from isolated worktree for verifier review."""

        return _run(["git", "diff", "--stat"], cwd=Path(worktree_path))

    def commit_worktree_changes(worktree_path: str, message: str) -> str:
        """Commit changes in debugger worktree when writes are enabled."""

        if not settings.enable_git_write:
            return json.dumps({"status": "blocked", "reason": "git writes disabled"})
        cwd = Path(worktree_path)
        _run(["git", "add", "-A"], cwd=cwd)
        output = _run(["git", "commit", "-m", message], cwd=cwd)
        sha = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
        return json.dumps({"status": "committed", "commit_sha": sha.strip(), "output": output})

    def create_pull_request(worktree_path: str, title: str, body: str, base_branch: str | None = None) -> str:
        """Open PR through gh CLI.

        Default off because PR creation crosses repo boundary and needs auth.
        """

        if not settings.enable_prs:
            return json.dumps({"status": "blocked", "reason": "PR creation disabled"})
        args = ["gh", "pr", "create", "--title", title, "--body", body]
        if base_branch:
            args.extend(["--base", base_branch])
        if settings.github_repo:
            args.extend(["--repo", settings.github_repo])
        output = _run(args, cwd=Path(worktree_path))
        return json.dumps({"status": "opened", "output": output})

    shared = [
        sdk.function_tool(read_context_snapshot),
        sdk.function_tool(read_repo_file),
        sdk.function_tool(list_repo_files),
        sdk.function_tool(run_readonly_repo_command),
    ]
    tester = list(shared)
    critic = list(shared)
    verifier = list(shared) + [
        sdk.function_tool(read_worktree_diff),
    ]
    debugger = list(shared) + [
        sdk.function_tool(create_branch_worktree),
        sdk.function_tool(commit_worktree_changes),
        sdk.function_tool(create_pull_request),
        sdk.function_tool(read_worktree_diff),
    ]
    return ToolGroups(shared=shared, tester=tester, critic=critic, debugger=debugger, verifier=verifier)
