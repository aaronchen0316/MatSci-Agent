from __future__ import annotations

import json
import subprocess
from pathlib import Path

import multiagent.tools as harness_tools
from multiagent.schemas import LiveEvalEvidence, LiveEvalInput
from multiagent.settings import MultiAgentSettings
from multiagent.tools import build_tool_groups, cleanup_worktree, create_target_base_worktree
from matsci_agent.schemas import DiscoveryConstraints


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
    (repo / "tests" / "sample.py").write_text("def test_sample():\n    assert True\n")
    (repo / "agent_specs" / "sample.md").write_text("# Sample\n")
    _run(["git", "init", "-b", "multi-agent"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    _run(["git", "update-ref", "refs/remotes/origin/multi-agent", "HEAD"], cwd=repo)
    return repo


def _settings(repo: Path, tmp_path: Path, enable_git_write: bool = True) -> MultiAgentSettings:
    return MultiAgentSettings(
        tool_root=repo, target_repo=repo,
        enable_git_write=enable_git_write,
        target_base_branch="multi-agent",
        worktree_root=tmp_path / "worktrees",
    )


def test_worktree_edit_diff_and_commit_flow(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)

    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-1"))
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
    assert "-VALUE = 1" in tools["read_worktree_patch"](worktree_path)
    assert "+VALUE = 2" in tools["read_worktree_patch"](worktree_path)


def test_debugger_pytest_tool_installs_dev_extra(monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-dev-extra"))
    real_run = harness_tools._run_completed
    commands: list[list[str]] = []

    def run(args, cwd):
        if args[:2] == ["uv", "run"]:
            commands.append(args)
            return subprocess.CompletedProcess(args, 0, "1 passed", "")
        return real_run(args, cwd)

    monkeypatch.setattr(harness_tools, "_run_completed", run)

    assert tools["run_worktree_pytest"](created["worktree_path"], ["tests/sample.py"]) == "1 passed"
    assert commands == [["uv", "run", "--extra", "dev", "pytest", "-q", "--", "tests/sample.py"]]


def test_worktree_creation_allocates_safe_suffix_for_retained_branch_and_path(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)

    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-2"))
    assert created["status"] == "created"

    retried = json.loads(tools["create_branch_worktree"]("fix/retrieval-2"))
    assert retried["status"] == "created"
    assert retried["branch_name"] == "fix/retrieval-2-2"


def test_mutation_tool_rejects_disallowed_path(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-3"))

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
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-4"))

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


def test_mutation_tool_rejects_traversal_hidden_by_allowed_prefix(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-prefix-traversal"))

    result = json.loads(
        tools["apply_worktree_text_edit"](
            created["worktree_path"],
            "src/matsci_agent/../../README.md",
            "append",
            insert_text="oops\n",
        )
    )

    assert result["status"] == "error"
    assert "allowlisted" in result["details"]


def test_mutation_tool_rejects_unsupported_suffix(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-5"))

    result = json.loads(
        tools["apply_worktree_text_edit"](
            created["worktree_path"],
            "src/matsci_agent/data.csv",
            "append",
            insert_text="3,4\n",
        )
    )
    assert result["status"] == "error"
    assert "allowlisted" in result["details"]


def test_mutation_tool_returns_error_for_missing_anchor_or_old_text(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-6"))

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


def test_mutation_tool_rejects_tooling_prompt_file(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-7"))
    worktree_path = created["worktree_path"]

    result = json.loads(
        tools["apply_worktree_text_edit"](
            worktree_path,
            "agent_specs/sample.md",
            "append",
            insert_text="\nextra line\n",
        )
    )
    assert result["status"] == "error"
    assert "allowlisted" in result["details"]


def test_commit_returns_blocked_when_no_changes_exist(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-8"))

    committed = json.loads(tools["commit_worktree_changes"](created["worktree_path"], "no changes"))
    assert committed["status"] == "blocked"
    assert committed["reason"] == "no changes"


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

    assert tester_names == {"run_live_retrieval_eval"}
    assert "apply_worktree_text_edit" not in tester_names
    assert "read_worktree_patch" not in tester_names
    assert critic_names == set()
    assert "run_worktree_pytest" in debugger_names
    assert "read_target_file" in debugger_names
    assert "run_live_retrieval_eval" not in verifier_names
    assert "run_readonly_repo_command" not in debugger_names
    assert "create_pull_request" not in debugger_names


def test_debugger_pytest_tool_rejects_shell_like_targets(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-pytest"))

    for target in [
        "tests/sample.py; touch /tmp/pwned",
        "tests/../src/matsci_agent/module.py",
        "../tests/sample.py",
        "src/module.py",
    ]:
        try:
            tools["run_worktree_pytest"](created["worktree_path"], [target])
        except ValueError as exc:
            assert "not allowed" in str(exc)
        else:
            raise AssertionError(f"target unexpectedly accepted: {target}")


def test_branch_name_rejects_traversal_and_absolute_paths(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)

    for branch_name in [
        "../escape",
        "nested/branch",
        "/tmp/escape",
        "retrieval..fix",
        "UPPER",
        "head",
        "repair.lock",
        "repair;touch-pwned",
        "$(touch-pwned)",
    ]:
        result = json.loads(tools["create_branch_worktree"](branch_name))
        assert result["status"] == "error"

    assert json.loads(tools["create_branch_worktree"]("fix/valid-repair"))["status"] == "created"


def test_worktree_tools_reject_unregistered_directory_and_symlink_escape(tmp_path: Path):
    repo = _make_repo(tmp_path)
    settings = _settings(repo, tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), settings).debugger)
    rogue = settings.worktree_root / "rogue"
    (rogue / "src" / "matsci_agent").mkdir(parents=True)
    (rogue / "src" / "matsci_agent" / "module.py").write_text("VALUE = 2\n")

    try:
        tools["read_worktree_file"](str(rogue), "src/matsci_agent/module.py")
    except ValueError as exc:
        assert "registered git worktree" in str(exc)
    else:
        raise AssertionError("unregistered directory unexpectedly accepted")

    escaped = settings.worktree_root / "escaped"
    escaped.parent.mkdir(parents=True, exist_ok=True)
    escaped.symlink_to(repo, target_is_directory=True)
    try:
        tools["read_worktree_file"](str(escaped), "src/matsci_agent/module.py")
    except ValueError as exc:
        assert "escapes configured root" in str(exc)
    else:
        raise AssertionError("symlink escape unexpectedly accepted")


def test_scoped_live_evaluator_executes_from_active_target_root(monkeypatch, tmp_path: Path):
    repo = _make_repo(tmp_path)
    tool_root = tmp_path / "tooling"
    (tool_root / "src").mkdir(parents=True)
    settings = MultiAgentSettings(
        tool_root=tool_root,
        target_repo=repo,
        active_target_root=repo,
        target_base_branch="multi-agent",
        worktree_root=tmp_path / "worktrees",
    )
    observed: dict[str, object] = {}
    expected = LiveEvalEvidence(status="blocked", query="find oxides", blocked_reason="fixture")

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed.update(kwargs)
        return subprocess.CompletedProcess(args, 0, expected.model_dump_json(), "")

    monkeypatch.setattr(harness_tools.subprocess, "run", fake_run)

    result = harness_tools._run_scoped_live_evaluation(
        settings,
        LiveEvalInput(
            query="find oxides",
            constraints=DiscoveryConstraints(),
            allow_live_mp=True,
        ),
    )

    assert result == expected
    assert observed["args"] == [harness_tools.sys.executable, "-m", "multiagent.scoped_evaluator"]
    assert observed["cwd"] == str(repo)
    assert str(observed["env"]["PYTHONPATH"]).split(":")[:2] == [
        str((repo / "src").resolve()),
        str((tool_root / "src").resolve()),
    ]
    assert '"allow_live_mp":true' in str(observed["input"])
    assert '"max_energy_above_hull"' not in str(observed["input"])


def test_target_base_worktree_comes_from_configured_product_branch(tmp_path: Path):
    repo = _make_repo(tmp_path)
    _run(["git", "branch", "main"], repo)
    _run(["git", "checkout", "main"], repo)
    Path(repo, "src/matsci_agent/module.py").write_text("VALUE = 3\n")
    _run(["git", "commit", "-am", "main value"], repo)
    _run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], repo)
    _run(["git", "checkout", "multi-agent"], repo)
    settings = _settings(repo, tmp_path)
    settings = MultiAgentSettings(
        tool_root=settings.tool_root,
        target_repo=settings.target_repo,
        target_base_branch="main",
        worktree_root=settings.worktree_root,
    )

    created = create_target_base_worktree(settings)

    assert created["status"] == "created"
    base = Path(created["worktree_path"])
    assert (base / "src/matsci_agent/module.py").read_text() == "VALUE = 3\n"
    assert cleanup_worktree(settings, str(base))["status"] == "removed"


def test_commit_rejects_unallowlisted_changed_files(tmp_path: Path):
    repo = _make_repo(tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), _settings(repo, tmp_path)).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-unsafe"))
    worktree_path = Path(created["worktree_path"])
    (worktree_path / "README.md").write_text("unexpected change\n")

    committed = json.loads(tools["commit_worktree_changes"](str(worktree_path), "unsafe"))

    assert committed["status"] == "error"
    assert committed["reason"] == "changed files are not allowlisted"
    assert committed["files"] == ["README.md"]


def test_cleanup_removes_clean_worktree_and_retains_branch(tmp_path: Path):
    repo = _make_repo(tmp_path)
    settings = _settings(repo, tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), settings).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-cleanup"))

    cleanup = cleanup_worktree(settings, created["worktree_path"])

    assert cleanup["status"] == "removed"
    assert not Path(created["worktree_path"]).exists()
    assert _run(["git", "show-ref", "--verify", "refs/heads/fix/retrieval-cleanup"], repo).returncode == 0


def test_cleanup_refuses_dirty_worktree(tmp_path: Path):
    repo = _make_repo(tmp_path)
    settings = _settings(repo, tmp_path)
    tools = _tool_map(build_tool_groups(FakeSDK(), settings).debugger)
    created = json.loads(tools["create_branch_worktree"]("fix/retrieval-dirty"))
    Path(created["worktree_path"], "src/matsci_agent/module.py").write_text("VALUE = 9\n")

    cleanup = cleanup_worktree(settings, created["worktree_path"])

    assert cleanup["status"] == "blocked"
    assert Path(created["worktree_path"]).exists()
