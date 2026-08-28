from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from multiagent.schemas import LiveEvalEvidence, LiveEvalInput
from multiagent.settings import MultiAgentSettings

_BRANCH_SEGMENT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_RESERVED_BRANCH_NAMES = {"head", "fetch_head", "orig_head"}
_SCOPED_EVAL_TIMEOUT_SECONDS = 180
_MUTABLE_PATH_PREFIXES = (
    "src/matsci_agent/",
    "tests/",
)
_MUTABLE_SUFFIXES = {".py"}


@dataclass(frozen=True)
class ToolGroups:
    tester: list[object]
    critic: list[object]
    debugger: list[object]
    verifier: list[object]


def _run_completed(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def _validate_branch_name(branch_name: str, *, require_fix_namespace: bool = False) -> str:
    parts = branch_name.split("/")
    if not parts or any(
        not _BRANCH_SEGMENT_PATTERN.fullmatch(part)
        or part in {".", ".."}
        or ".." in part
        or part in _RESERVED_BRANCH_NAMES
        or part.endswith((".", ".lock"))
        for part in parts
    ):
        raise ValueError("branch_name must contain safe Git path segments")
    if require_fix_namespace and (len(parts) != 2 or parts[0] != "fix"):
        raise ValueError("repair branch_name must use fix/<issue>")
    return branch_name


def _resolve_under(root: Path, candidate: str, *, require_exists: bool = False) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {candidate}")
    resolved = (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"path escapes configured root: {candidate}")
    if require_exists and not resolved.exists():
        raise ValueError(f"path does not exist: {candidate}")
    return resolved


def _normalized_relative_path(relative_path: str) -> str | None:
    if not relative_path:
        return None
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = path.as_posix()
    return normalized if normalized not in {"", "."} else None


def _is_under(root: Path, path: Path) -> bool:
    return root == path or root in path.parents


def _registered_worktree_paths(settings: MultiAgentSettings) -> set[Path]:
    result = _run_completed(["git", "worktree", "list", "--porcelain"], settings.resolved_target_repo)
    if result.returncode != 0:
        raise ValueError("unable to list registered git worktrees")
    return {
        Path(line.removeprefix("worktree ")).resolve()
        for line in (result.stdout or "").splitlines()
        if line.startswith("worktree ")
    }


def _worktree_dir(settings: MultiAgentSettings, worktree_path: str) -> Path:
    root = settings.worktree_root.resolve()
    path = Path(worktree_path).resolve()
    if not _is_under(root, path) or root == path:
        raise ValueError(f"worktree path escapes configured root: {worktree_path}")
    if not path.is_dir():
        raise ValueError(f"worktree path does not exist: {worktree_path}")
    if path not in _registered_worktree_paths(settings):
        raise ValueError(f"worktree path is not a registered git worktree: {worktree_path}")
    return path


def _is_mutable_relative_path(relative_path: str) -> bool:
    normalized = _normalized_relative_path(relative_path)
    if normalized is None:
        return False
    path = Path(normalized)
    return (
        any(normalized.startswith(prefix) for prefix in _MUTABLE_PATH_PREFIXES)
        and path.suffix in _MUTABLE_SUFFIXES
    )


def _worktree_file(settings: MultiAgentSettings, worktree_path: str, relative_path: str) -> Path:
    if not _is_mutable_relative_path(relative_path):
        raise ValueError(f"path not allowlisted for mutation: {relative_path}")
    worktree_dir = _worktree_dir(settings, worktree_path)
    path = _resolve_under(worktree_dir, relative_path, require_exists=True)
    if not path.is_file():
        raise ValueError(f"file does not exist: {relative_path}")
    return path


def _target_file(settings: MultiAgentSettings, relative_path: str) -> Path:
    if not _is_mutable_relative_path(relative_path):
        raise ValueError(f"path not allowlisted for product inspection: {relative_path}")
    path = _resolve_under(settings.resolved_target_root, relative_path, require_exists=True)
    if not path.is_file():
        raise ValueError(f"file does not exist: {relative_path}")
    return path


def _test_targets(root: Path, targets: list[str]) -> list[str]:
    if not targets or len(targets) > 10:
        raise ValueError("provide between 1 and 10 test targets")
    tests_root = (root / "tests").resolve()
    normalized: list[str] = []
    for target in targets:
        normalized_target = _normalized_relative_path(target)
        if normalized_target is None or not normalized_target.startswith("tests/") or Path(normalized_target).suffix != ".py":
            raise ValueError(f"test target not allowed: {target}")
        path = _resolve_under(root, normalized_target, require_exists=True)
        if not path.is_file() or not _is_under(tests_root, path) or path == tests_root:
            raise ValueError(f"test target is not a file: {target}")
        normalized.append(str(path.relative_to(root)))
    return normalized


def _numbered_slice(path: Path, start_line: int = 1, end_line: int = 240) -> str:
    lines = path.read_text().splitlines()
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line))
    return "\n".join(f"{index}: {line}" for index, line in enumerate(lines[start - 1 : end], start=start))


def worktree_evidence(settings: MultiAgentSettings, worktree_path: str) -> dict[str, str]:
    cwd = _worktree_dir(settings, worktree_path)
    return {
        "status": _output(_run_completed(["git", "status", "--short"], cwd)),
        "diff_stat": _output(_run_completed(["git", "diff", "--stat"], cwd)),
        "patch": _output(_run_completed(["git", "diff", "--"], cwd)),
        "head": _output(_run_completed(["git", "rev-parse", "HEAD"], cwd)),
    }


def cleanup_worktree(settings: MultiAgentSettings, worktree_path: str) -> dict[str, str]:
    """Remove a clean worktree while preserving its branch."""

    try:
        cwd = _worktree_dir(settings, worktree_path)
    except ValueError as exc:
        return {"status": "failed", "reason": str(exc)}

    dirty = _output(_run_completed(["git", "status", "--porcelain"], cwd))
    if dirty:
        return {"status": "blocked", "reason": "worktree has uncommitted changes"}
    result = _run_completed(["git", "worktree", "remove", str(cwd)], settings.resolved_target_repo)
    if result.returncode != 0:
        return {"status": "failed", "reason": "git worktree remove failed", "output": _output(result)}
    return {"status": "removed", "worktree_path": str(cwd)}


def create_target_base_worktree(settings: MultiAgentSettings) -> dict[str, str]:
    """Create a clean detached product checkout from the configured base."""

    root = settings.worktree_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".base-{uuid4().hex[:12]}"
    result = _run_completed(
        ["git", "worktree", "add", "--detach", str(path), settings.target_base_ref],
        settings.resolved_target_repo,
    )
    if result.returncode != 0:
        return {"status": "failed", "reason": _output(result)}
    return {"status": "created", "worktree_path": str(path.resolve())}


def _run_scoped_live_evaluation(settings: MultiAgentSettings, payload: LiveEvalInput) -> LiveEvalEvidence:
    env = os.environ.copy()
    target_source_root = str((settings.resolved_target_root / "src").resolve())
    tool_source_root = str((settings.resolved_tool_root / "src").resolve())
    inherited_pythonpath = env.get("PYTHONPATH")
    source_paths = [target_source_root, tool_source_root]
    if inherited_pythonpath:
        source_paths.append(inherited_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env["MATSCI_NLP_MODEL"] = settings.product_model
    if settings.base_url:
        env["MATSCI_LLM_BASE_URL"] = settings.base_url
    if settings.api_key:
        env["MULTIAGENT_PRODUCT_API_KEY"] = settings.api_key
        env["MATSCI_LLM_API_KEY_ENV"] = "MULTIAGENT_PRODUCT_API_KEY"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "multiagent.scoped_evaluator"],
            cwd=str(settings.resolved_target_root),
            input=payload.model_dump_json(exclude_none=True),
            capture_output=True,
            text=True,
            check=False,
            timeout=_SCOPED_EVAL_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return LiveEvalEvidence(status="blocked", query=payload.query, blocked_reason="scoped evaluator timed out")
    except OSError as exc:
        return LiveEvalEvidence(status="blocked", query=payload.query, blocked_reason=f"scoped evaluator failed: {type(exc).__name__}")

    try:
        return LiveEvalEvidence.model_validate_json(result.stdout)
    except Exception:
        return LiveEvalEvidence(
            status="blocked",
            query=payload.query,
            blocked_reason=f"scoped evaluator returned invalid output (exit {result.returncode})",
        )


def run_scoped_live_evaluation(settings: MultiAgentSettings, payload: LiveEvalInput) -> LiveEvalEvidence:
    """Run typed live evaluation from the supplied checkout."""

    return _run_scoped_live_evaluation(settings, payload)


def build_tool_groups(sdk, settings: MultiAgentSettings) -> ToolGroups:
    """Create narrow typed tool surfaces for each specialist agent."""

    def read_target_file(relative_path: str, start_line: int = 1, end_line: int = 240) -> str:
        """Read one product or test file slice from the active product checkout."""

        path = _target_file(settings, relative_path)
        return _numbered_slice(path, start_line=start_line, end_line=end_line)

    def create_branch_worktree(branch_name: str) -> str:
        """Create a validated isolated worktree for debugger changes."""

        if not settings.enable_git_write:
            return json.dumps({"status": "blocked", "reason": "git writes disabled"})
        try:
            safe_branch = _validate_branch_name(branch_name, require_fix_namespace=True)
        except ValueError as exc:
            return json.dumps({"status": "error", "reason": str(exc), "branch_name": branch_name})

        root = settings.worktree_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        selected_branch = safe_branch
        worktree_path = _resolve_under(root, selected_branch)
        for suffix in range(2, 100):
            branch_check = _run_completed(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{selected_branch}"],
                settings.resolved_target_repo,
            )
            if not worktree_path.exists() and branch_check.returncode != 0:
                break
            try:
                selected_branch = _validate_branch_name(f"{safe_branch}-{suffix}", require_fix_namespace=True)
            except ValueError:
                return json.dumps(
                    {
                        "status": "blocked",
                        "reason": "no safe unique repair branch name is available",
                        "branch_name": safe_branch,
                    }
                )
            worktree_path = _resolve_under(root, selected_branch)
        else:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "no safe unique repair branch name is available",
                    "branch_name": safe_branch,
                }
            )
        result = _run_completed(
            ["git", "worktree", "add", "-b", selected_branch, str(worktree_path), settings.target_base_ref],
            settings.resolved_target_repo,
        )
        if result.returncode != 0:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "git worktree add failed",
                    "branch_name": selected_branch,
                    "worktree_path": str(worktree_path),
                    "output": _output(result),
                }
            )
        return json.dumps(
            {
                "status": "created",
                "branch_name": selected_branch,
                "worktree_path": str(worktree_path),
                "output": _output(result),
            }
        )

    def read_worktree_file(worktree_path: str, relative_path: str, start_line: int = 1, end_line: int = 240) -> str:
        """Read an allowlisted worktree text file slice."""

        return _numbered_slice(_worktree_file(settings, worktree_path, relative_path), start_line, end_line)

    def read_worktree_diff(worktree_path: str) -> str:
        """Read isolated-worktree diff stat."""

        return _output(_run_completed(["git", "diff", "--stat"], _worktree_dir(settings, worktree_path)))

    def read_worktree_patch(worktree_path: str) -> str:
        """Read isolated-worktree full patch."""

        return _output(_run_completed(["git", "diff", "--"], _worktree_dir(settings, worktree_path)))

    def run_worktree_pytest(worktree_path: str, targets: list[str]) -> str:
        """Run bounded changed-test file targets from a managed repair worktree."""

        worktree = _worktree_dir(settings, worktree_path)
        normalized = _test_targets(worktree, targets)
        return _output(_run_completed(["uv", "run", "pytest", "-q", "--", *normalized], worktree))

    def apply_worktree_text_edit(
        worktree_path: str,
        relative_path: str,
        op: str,
        anchor_text: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        insert_text: str | None = None,
    ) -> str:
        """Apply one bounded edit to an existing allowlisted worktree file."""

        if not settings.enable_git_write:
            return json.dumps({"status": "blocked", "relative_path": relative_path, "operation": op, "details": "git writes disabled"})
        try:
            path = _worktree_file(settings, worktree_path, relative_path)
            original = path.read_text()
            if op == "replace_once":
                if old_text is None or new_text is None:
                    raise ValueError("replace_once requires old_text and new_text")
                if old_text not in original:
                    raise ValueError("old_text not found")
                updated = original.replace(old_text, new_text, 1)
            elif op == "insert_after":
                if anchor_text is None or insert_text is None:
                    raise ValueError("insert_after requires anchor_text and insert_text")
                if anchor_text not in original:
                    raise ValueError("anchor_text not found")
                insert_at = original.index(anchor_text) + len(anchor_text)
                updated = original[:insert_at] + insert_text + original[insert_at:]
            elif op == "append":
                if insert_text is None:
                    raise ValueError("append requires insert_text")
                updated = original + insert_text
            else:
                raise ValueError(f"unsupported operation: {op}")
            if updated == original:
                return json.dumps({"status": "no_change", "relative_path": relative_path, "operation": op, "details": "edit produced no change"})
            path.write_text(updated)
            return json.dumps({"status": "patched", "relative_path": relative_path, "operation": op, "details": "edit applied"})
        except ValueError as exc:
            return json.dumps({"status": "error", "relative_path": relative_path, "operation": op, "details": str(exc)})

    def commit_worktree_changes(worktree_path: str, message: str) -> str:
        """Commit only changed allowlisted files in an isolated worktree."""

        if not settings.enable_git_write:
            return json.dumps({"status": "blocked", "reason": "git writes disabled"})
        try:
            cwd = _worktree_dir(settings, worktree_path)
        except ValueError as exc:
            return json.dumps({"status": "error", "reason": str(exc)})
        changed = {
            path
            for path in _run_completed(["git", "diff", "--name-only", "-z"], cwd).stdout.split("\0")
            if path
        }
        changed.update(
            path
            for path in _run_completed(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd).stdout.split("\0")
            if path
        )
        if not changed:
            return json.dumps({"status": "blocked", "reason": "no changes"})
        disallowed = sorted(path for path in changed if not _is_mutable_relative_path(path))
        if disallowed:
            return json.dumps({"status": "error", "reason": "changed files are not allowlisted", "files": disallowed})
        add_result = _run_completed(["git", "add", "--", *sorted(changed)], cwd)
        if add_result.returncode != 0:
            return json.dumps({"status": "error", "reason": "git add failed", "output": _output(add_result)})
        staged_check = _run_completed(["git", "diff", "--cached", "--quiet"], cwd)
        if staged_check.returncode == 0:
            return json.dumps({"status": "blocked", "reason": "no staged changes"})
        commit_result = _run_completed(["git", "commit", "-m", message], cwd)
        if commit_result.returncode != 0:
            return json.dumps({"status": "error", "reason": "git commit failed", "output": _output(commit_result)})
        sha = _output(_run_completed(["git", "rev-parse", "HEAD"], cwd))
        return json.dumps({"status": "committed", "commit_sha": sha, "output": _output(commit_result)})

    def run_live_retrieval_eval(payload: LiveEvalInput) -> dict[str, object]:
        """Run one typed live retrieval evaluation in the active checkout."""

        return _run_scoped_live_evaluation(settings, payload).model_dump(mode="json")

    tester = [sdk.function_tool(run_live_retrieval_eval)]
    critic: list[object] = []
    verifier = [
        sdk.function_tool(read_worktree_file),
        sdk.function_tool(read_worktree_patch),
    ]
    debugger = [
        sdk.function_tool(read_target_file),
        sdk.function_tool(create_branch_worktree),
        sdk.function_tool(read_worktree_file),
        sdk.function_tool(apply_worktree_text_edit),
        sdk.function_tool(run_worktree_pytest),
        sdk.function_tool(commit_worktree_changes),
        sdk.function_tool(read_worktree_diff),
        sdk.function_tool(read_worktree_patch),
    ]
    return ToolGroups(tester=tester, critic=critic, debugger=debugger, verifier=verifier)
