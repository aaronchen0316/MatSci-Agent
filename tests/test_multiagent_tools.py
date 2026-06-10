from __future__ import annotations

import json
import subprocess
from pathlib import Path

from matsci_agent.multiagent.settings import MultiAgentSettings
from matsci_agent.multiagent.tools import build_tool_groups


class FakeSDK:
    @staticmethod
    def function_tool(fn):
        return fn


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _tool_map(tools: list[object]) -> dict[str, object]:
    return {tool.__name__: tool for tool in tools}


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "matsci_agent").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "agent_specs").mkdir()
    (repo / "CONTEXT.md").write_text("# Context\n")
    (repo / "README.md").write_text("# Readme\n")
    (repo / "src" / "matsci_agent" / "module.py").write_text("VALUE = 1\n\ndef answer():\n    return VALUE\n")
    (repo / "src" / "matsci_agent" / "data.csv").write_text("a,b\n1,2\n")
    (repo / "tests" / "sample.txt").write_text("alpha\n")
    (repo / "agent_specs" / "sample.md").write_text("# Sample\n")
    _run(["git", "init", "-b", "multi-agent"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    return repo


def _settings(repo: Path, tmp_path: Path, enable_git_write: bool = True) -> MultiAgentSettings:
    return MultiAgentSettings(
        repo_root=repo,
        enable_git_write=enable_git_write,
        enable_prs=False,
        base_branch="multi-agent",
        worktree_root=tmp_path / "worktrees",
    )


def test_worktree_edit_diff_and_commit_flow(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)

    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-1"))
    assert created["status"] == "created"
    worktree_path = created["worktree_path"]

    file_view = tools["read_worktree_file"](worktree_path, "src/matsci_agent/module.py")
    assert "1: VALUE = 1" in file_view

    patched = json.loads(
        tools["apply_worktree_text_edit"](
            worktree_path,
            "src/matsci_agent/module.py",
            "replace_once",
            old_text="VALUE = 1",
            new_text="VALUE = 2",
        )
    )
    assert patched["status"] == "patched"

    diff_stat = tools["read_worktree_diff"](worktree_path)
    assert "src/matsci_agent/module.py" in diff_stat

    patch = tools["read_worktree_patch"](worktree_path)
    assert "-VALUE = 1" in patch
    assert "+VALUE = 2" in patch

    committed = json.loads(tools["commit_worktree_changes"](worktree_path, "update value"))
    assert committed["status"] == "committed"
    assert len(committed["commit_sha"]) == 40


def test_worktree_creation_fails_fast_on_existing_branch_and_path(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)

    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-2"))
    assert created["status"] == "created"

    blocked = json.loads(tools["create_branch_worktree"]("retrieval-fix-2"))
    assert blocked["status"] == "blocked"
    assert blocked["reason"] in {"worktree path already exists", "branch already exists"}


def test_mutation_tool_rejects_disallowed_path(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-3"))

    result = json.loads(
        tools["apply_worktree_text_edit"](
            created["worktree_path"],
            "README.md",
            "append",
            insert_text="more\n",
        )
    )
    assert result["status"] == "error"
    assert "allowlisted" in result["details"]


def test_mutation_tool_rejects_path_traversal(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-4"))

    result = json.loads(
        tools["apply_worktree_text_edit"](
            created["worktree_path"],
            "../src/matsci_agent/module.py",
            "append",
            insert_text="oops\n",
        )
    )
    assert result["status"] == "error"
    assert "allowlisted" in result["details"]


def test_mutation_tool_rejects_unsupported_suffix(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-5"))

    result = json.loads(
        tools["apply_worktree_text_edit"](
            created["worktree_path"],
            "src/matsci_agent/data.csv",
            "append",
            insert_text="3,4\n",
        )
    )
    assert result["status"] == "error"
    assert "file type not allowed" in result["details"]


def test_mutation_tool_returns_error_for_missing_anchor_or_old_text(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-6"))

    result = json.loads(
        tools["apply_worktree_text_edit"](
            created["worktree_path"],
            "src/matsci_agent/module.py",
            "replace_once",
            old_text="VALUE = 999",
            new_text="VALUE = 2",
        )
    )
    assert result["status"] == "error"
    assert result["details"] == "old_text not found"


def test_append_edit_works_on_existing_allowlisted_text_file(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-7"))
    worktree_path = created["worktree_path"]

    result = json.loads(
        tools["apply_worktree_text_edit"](
            worktree_path,
            "agent_specs/sample.md",
            "append",
            insert_text="\nextra line\n",
        )
    )
    assert result["status"] == "patched"

    content = (Path(worktree_path) / "agent_specs" / "sample.md").read_text()
    assert content.endswith("\nextra line\n")


def test_commit_returns_blocked_when_no_changes_exist(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("retrieval-fix-8"))

    committed = json.loads(tools["commit_worktree_changes"](created["worktree_path"], "no changes"))
    assert committed["status"] == "blocked"
    assert committed["reason"] == "no staged changes"


def test_tool_groups_keep_mutation_limited_to_debugger(tmp_path: Path):
    repo = _make_repo(tmp_path)
    groups = build_tool_groups(FakeSDK(), _settings(repo, tmp_path))

    debugger_names = {tool.__name__ for tool in groups.debugger}
    verifier_names = {tool.__name__ for tool in groups.verifier}
    tester_names = {tool.__name__ for tool in groups.tester}
    critic_names = {tool.__name__ for tool in groups.critic}

    assert "apply_worktree_text_edit" in debugger_names
    assert "read_worktree_patch" in debugger_names
    assert "read_worktree_file" in debugger_names

    assert "apply_worktree_text_edit" not in verifier_names
    assert "read_worktree_patch" in verifier_names
    assert "read_worktree_file" in verifier_names

    assert "run_live_retrieval_eval" in tester_names
    assert "apply_worktree_text_edit" not in tester_names
    assert "read_worktree_patch" not in tester_names
    assert "run_live_retrieval_eval" not in critic_names
    assert "apply_worktree_text_edit" not in critic_names
    assert "read_worktree_patch" not in critic_names
