from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from matsci_agent.multiagent.evaluator import LiveRetrievalEvaluator
from matsci_agent.multiagent.schemas import LiveEvalInput
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

_MUTABLE_PATH_PREFIXES = (
    "src/matsci_agent/",
    "tests/",
    "agent_specs/",
)

_MUTABLE_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
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

    def _run_completed(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(cwd or settings.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def _run(args: list[str], cwd: Path | None = None) -> str:
        result = _run_completed(args, cwd=cwd)
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()

    def _readonly_allowed(argv: list[str]) -> bool:
        for prefix in _READONLY_PREFIXES:
            if tuple(argv[: len(prefix)]) == prefix:
                return True
        return False

    def _worktree_dir(worktree_path: str) -> Path:
        root = settings.worktree_root.resolve()
        path = Path(worktree_path).resolve()
        if root not in path.parents:
            raise ValueError(f"worktree path escapes configured root: {worktree_path}")
        if not path.exists() or not path.is_dir():
            raise ValueError(f"worktree path does not exist: {worktree_path}")
        return path

    def _worktree_file(worktree_path: str, relative_path: str) -> Path:
        if not relative_path:
            raise ValueError("relative_path is required")
        if Path(relative_path).is_absolute():
            raise ValueError(f"absolute paths are not allowed: {relative_path}")
        if not any(relative_path.startswith(prefix) for prefix in _MUTABLE_PATH_PREFIXES):
            raise ValueError(f"path not allowlisted for mutation: {relative_path}")

        worktree_dir = _worktree_dir(worktree_path)
        path = (worktree_dir / relative_path).resolve()
        if worktree_dir not in path.parents:
            raise ValueError(f"path escapes worktree root: {relative_path}")
        if not path.exists() or not path.is_file():
            raise ValueError(f"file does not exist: {relative_path}")
        if path.suffix not in _MUTABLE_SUFFIXES:
            raise ValueError(f"file type not allowed: {path.suffix}")
        return path

    def _numbered_slice(path: Path, start_line: int = 1, end_line: int = 240) -> str:
        lines = path.read_text().splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        selected = lines[start - 1 : end]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start)]
        return "\n".join(numbered)

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
        return _numbered_slice(path, start_line=start_line, end_line=end_line)

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
        worktree_path = (settings.worktree_root / branch_name).resolve()
        if worktree_path.exists():
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "worktree path already exists",
                    "branch_name": branch_name,
                    "worktree_path": str(worktree_path),
                }
            )
        branch_check = _run_completed(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            cwd=settings.repo_root,
        )
        if branch_check.returncode == 0:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "branch already exists",
                    "branch_name": branch_name,
                    "worktree_path": str(worktree_path),
                }
            )
        result = _run_completed(
            [
                "git",
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_path),
                settings.base_branch,
            ],
            cwd=settings.repo_root,
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "git worktree add failed",
                    "branch_name": branch_name,
                    "worktree_path": str(worktree_path),
                    "output": output,
                }
            )
        return json.dumps({"status": "created", "branch_name": branch_name, "worktree_path": str(worktree_path), "output": output})

    def read_worktree_file(worktree_path: str, relative_path: str, start_line: int = 1, end_line: int = 240) -> str:
        """Read numbered file slice from isolated worktree."""

        path = _worktree_file(worktree_path, relative_path)
        return _numbered_slice(path, start_line=start_line, end_line=end_line)

    def read_worktree_diff(worktree_path: str) -> str:
        """Read diff from isolated worktree for verifier review."""

        return _run(["git", "diff", "--stat"], cwd=_worktree_dir(worktree_path))

    def read_worktree_patch(worktree_path: str) -> str:
        """Read full unified diff from isolated worktree."""

        return _run(["git", "diff", "--"], cwd=_worktree_dir(worktree_path))

    def apply_worktree_text_edit(
        worktree_path: str,
        relative_path: str,
        op: str,
        anchor_text: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        insert_text: str | None = None,
    ) -> str:
        """Apply one bounded text edit inside isolated worktree.

        Only existing allowlisted text files may be modified.
        """

        if not settings.enable_git_write:
            return json.dumps({"status": "blocked", "relative_path": relative_path, "operation": op, "details": "git writes disabled"})

        try:
            path = _worktree_file(worktree_path, relative_path)
            original = path.read_text()
            updated = original

            if op == "replace_once":
                if old_text is None or new_text is None:
                    raise ValueError("replace_once requires old_text and new_text")
                if old_text not in updated:
                    raise ValueError("old_text not found")
                updated = updated.replace(old_text, new_text, 1)
            elif op == "insert_after":
                if anchor_text is None or insert_text is None:
                    raise ValueError("insert_after requires anchor_text and insert_text")
                if anchor_text not in updated:
                    raise ValueError("anchor_text not found")
                insert_at = updated.index(anchor_text) + len(anchor_text)
                updated = updated[:insert_at] + insert_text + updated[insert_at:]
            elif op == "append":
                if insert_text is None:
                    raise ValueError("append requires insert_text")
                updated = updated + insert_text
            else:
                raise ValueError(f"unsupported operation: {op}")

            if updated == original:
                return json.dumps(
                    {
                        "status": "no_change",
                        "relative_path": relative_path,
                        "operation": op,
                        "details": "edit produced no change",
                    }
                )

            path.write_text(updated)
            return json.dumps(
                {
                    "status": "patched",
                    "relative_path": relative_path,
                    "operation": op,
                    "details": "edit applied",
                }
            )
        except ValueError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "relative_path": relative_path,
                    "operation": op,
                    "details": str(exc),
                }
            )

    def commit_worktree_changes(worktree_path: str, message: str) -> str:
        """Commit changes in debugger worktree when writes are enabled."""

        if not settings.enable_git_write:
            return json.dumps({"status": "blocked", "reason": "git writes disabled"})
        cwd = _worktree_dir(worktree_path)
        add_result = _run_completed(["git", "add", "-A"], cwd=cwd)
        if add_result.returncode != 0:
            output = ((add_result.stdout or "") + (add_result.stderr or "")).strip()
            return json.dumps({"status": "error", "reason": "git add failed", "output": output})
        staged_check = _run_completed(["git", "diff", "--cached", "--quiet"], cwd=cwd)
        if staged_check.returncode == 0:
            return json.dumps({"status": "blocked", "reason": "no staged changes"})
        commit_result = _run_completed(["git", "commit", "-m", message], cwd=cwd)
        output = ((commit_result.stdout or "") + (commit_result.stderr or "")).strip()
        if commit_result.returncode != 0:
            return json.dumps({"status": "error", "reason": "git commit failed", "output": output})
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
        result = _run_completed(args, cwd=_worktree_dir(worktree_path))
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            return json.dumps({"status": "error", "reason": "gh pr create failed", "output": output})
        return json.dumps({"status": "opened", "output": output})

    live_evaluator = LiveRetrievalEvaluator()

    def run_live_retrieval_eval(objective: str, allow_live_mp: bool = False) -> dict[str, object]:
        """Run one typed live retrieval eval for Retrieval Tester."""

        evidence = live_evaluator.evaluate(
            LiveEvalInput(query=objective, allow_live_mp=allow_live_mp)
        )
        return evidence.model_dump(mode="json")

    shared = [
        sdk.function_tool(read_context_snapshot),
        sdk.function_tool(read_repo_file),
        sdk.function_tool(list_repo_files),
        sdk.function_tool(run_readonly_repo_command),
    ]
    tester = list(shared) + [
        sdk.function_tool(run_live_retrieval_eval),
    ]
    critic = list(shared)
    verifier = list(shared) + [
        sdk.function_tool(read_worktree_file),
        sdk.function_tool(read_worktree_diff),
        sdk.function_tool(read_worktree_patch),
    ]
    debugger = list(shared) + [
        sdk.function_tool(create_branch_worktree),
        sdk.function_tool(read_worktree_file),
        sdk.function_tool(apply_worktree_text_edit),
        sdk.function_tool(commit_worktree_changes),
        sdk.function_tool(create_pull_request),
        sdk.function_tool(read_worktree_diff),
        sdk.function_tool(read_worktree_patch),
    ]
    return ToolGroups(shared=shared, tester=tester, critic=critic, debugger=debugger, verifier=verifier)
