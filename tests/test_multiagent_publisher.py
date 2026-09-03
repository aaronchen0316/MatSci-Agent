from __future__ import annotations

import subprocess
from pathlib import Path

import multiagent.publisher as publisher
from multiagent.publisher import publish_and_merge_repair
from multiagent.settings import MultiAgentSettings


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, worktree_root=tmp_path / "worktrees")


def _branch_run(commands: list[list[str]], *, branch_sha: str = "head-sha"):
    def fake_run(args, _cwd):
        commands.append(args)
        if args[:3] == ["git", "rev-parse", "--verify"] and args[-1] == "fix/volume":
            return subprocess.CompletedProcess(args, 0, f"{branch_sha}\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    return fake_run


def test_publisher_requires_github_token(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = publish_and_merge_repair(_settings(tmp_path), branch_name="fix/volume", artifact_dir=tmp_path)

    assert result.status == "blocked"
    assert result.summary == "missing GITHUB_TOKEN"


def test_publisher_rejects_non_fix_branch_before_push(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")

    result = publish_and_merge_repair(_settings(tmp_path), branch_name="retrieval-fix-retry", artifact_dir=tmp_path)

    assert result.status == "blocked"
    assert "fix/<issue>" in result.summary


def test_publisher_requires_remote_tracking_base(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(publisher, "_run", lambda args, _cwd: subprocess.CompletedProcess(args, 0, "", ""))

    assert publisher._ensure_clean_tooling(settings) == "target base ref does not exist: origin/main"


def test_wait_for_checks_requires_named_product_ci_and_no_failed_checks(monkeypatch):
    monkeypatch.setattr(
        publisher,
        "_github_json",
        lambda **_kwargs: {
            "check_runs": [
                {"name": "Product CI / test", "status": "completed", "conclusion": "success"},
                {"name": "lint", "status": "completed", "conclusion": "success"},
            ]
        },
    )

    assert publisher._wait_for_checks(repository="example/repo", token="token", sha="head", timeout_seconds=1) == "pass"


def test_wait_for_checks_rejects_missing_required_or_failed_extra_check(monkeypatch):
    monkeypatch.setattr(
        publisher,
        "_github_json",
        lambda **_kwargs: {"check_runs": [{"name": "lint", "status": "completed", "conclusion": "success"}]},
    )
    assert publisher._wait_for_checks(repository="example/repo", token="token", sha="head", timeout_seconds=1) == "fail"
    monkeypatch.setattr(
        publisher,
        "_github_json",
        lambda **_kwargs: {
            "check_runs": [
                {"name": "Product CI / test", "status": "completed", "conclusion": "success"},
                {"name": "lint", "status": "completed", "conclusion": "failure"},
            ]
        },
    )
    assert publisher._wait_for_checks(repository="example/repo", token="token", sha="head", timeout_seconds=1) == "fail"


def test_publisher_rejects_non_product_diff_before_push(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_tooling", lambda *_args: None)
    monkeypatch.setattr(publisher, "_load_production_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher, "_product_only_diff", lambda *_args: "repair diff includes non-product paths: src/multiagent/tools.py")
    commands: list[list[str]] = []
    monkeypatch.setattr(publisher, "_run", _branch_run(commands))

    result = publish_and_merge_repair(_settings(tmp_path), branch_name="fix/volume", artifact_dir=tmp_path)

    assert result.status == "blocked"
    assert "non-product" in result.summary
    assert not any(command[:2] == ["git", "push"] for command in commands)


def test_publisher_pushes_ready_pr_waits_ci_and_squash_merges(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_tooling", lambda *_args: None)
    monkeypatch.setattr(publisher, "_load_production_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher, "_product_only_diff", lambda *_args: None)
    monkeypatch.setattr(publisher, "_run_branch_suite", lambda *_args: (True, "173 passed"))
    monkeypatch.setattr(publisher, "_github_repository", lambda *_args: "example/repo")
    monkeypatch.setattr(publisher, "_create_ready_pr", lambda **_kwargs: (42, "https://example.test/pr/42", "head-sha"))
    monkeypatch.setattr(publisher, "_wait_for_checks", lambda **_kwargs: "pass")
    observed: dict[str, object] = {}

    def merge(**kwargs):
        observed["merge"] = kwargs
        return "merge-sha"

    monkeypatch.setattr(publisher, "_squash_merge", merge)
    commands: list[list[str]] = []
    monkeypatch.setattr(publisher, "_run", _branch_run(commands))

    result = publish_and_merge_repair(_settings(tmp_path), branch_name="fix/volume", artifact_dir=tmp_path)

    assert result.status == "merged"
    assert result.head_sha == "head-sha"
    assert result.merge_sha == "merge-sha"
    assert observed["merge"]["sha"] == "head-sha"
    assert any(command[:2] == ["git", "push"] for command in commands)
    assert "secret" not in result.model_dump_json()


def test_publisher_retains_ready_pr_when_remote_ci_fails(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_tooling", lambda *_args: None)
    monkeypatch.setattr(publisher, "_load_production_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher, "_product_only_diff", lambda *_args: None)
    monkeypatch.setattr(publisher, "_run_branch_suite", lambda *_args: (True, "173 passed"))
    monkeypatch.setattr(publisher, "_github_repository", lambda *_args: "example/repo")
    monkeypatch.setattr(publisher, "_create_ready_pr", lambda **_kwargs: (42, "https://example.test/pr/42", "head-sha"))
    monkeypatch.setattr(publisher, "_wait_for_checks", lambda **_kwargs: "fail")
    commands: list[list[str]] = []
    monkeypatch.setattr(publisher, "_run", _branch_run(commands))
    merge_called = False

    def merge(**_kwargs):
        nonlocal merge_called
        merge_called = True
        return "merge-sha"

    monkeypatch.setattr(publisher, "_squash_merge", merge)

    result = publish_and_merge_repair(_settings(tmp_path), branch_name="fix/volume", artifact_dir=tmp_path)

    assert result.status == "published"
    assert result.ci_status == "fail"
    assert not merge_called


def test_publisher_never_merges_when_pr_head_differs_from_validated_branch(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_tooling", lambda *_args: None)
    monkeypatch.setattr(publisher, "_load_production_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher, "_product_only_diff", lambda *_args: None)
    monkeypatch.setattr(publisher, "_run_branch_suite", lambda *_args: (True, "173 passed"))
    monkeypatch.setattr(publisher, "_github_repository", lambda *_args: "example/repo")
    monkeypatch.setattr(publisher, "_create_ready_pr", lambda **_kwargs: (42, "https://example.test/pr/42", "unexpected-sha"))
    commands: list[list[str]] = []
    monkeypatch.setattr(publisher, "_run", _branch_run(commands))
    merge_called = False

    def merge(**_kwargs):
        nonlocal merge_called
        merge_called = True
        return "merge-sha"

    monkeypatch.setattr(publisher, "_squash_merge", merge)

    result = publish_and_merge_repair(_settings(tmp_path), branch_name="fix/volume", artifact_dir=tmp_path)

    assert result.status == "published"
    assert "does not match" in result.summary
    assert not merge_called
