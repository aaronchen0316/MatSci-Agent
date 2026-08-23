from __future__ import annotations

import subprocess
from pathlib import Path

import matsci_agent.multiagent.publisher as publisher
from matsci_agent.multiagent.publisher import publish_pull_request
from matsci_agent.multiagent.settings import MultiAgentSettings


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(repo_root=tmp_path, base_branch="multi-agent", worktree_root=tmp_path / "worktrees")


def test_publisher_requires_github_token(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = publish_pull_request(_settings(tmp_path), branch_name="retrieval-fix-1", base_branch="multi-agent")

    assert result.status == "blocked"
    assert result.summary == "missing GITHUB_TOKEN"


def test_validation_only_publication_requires_reason(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_base", lambda *_args: None)
    monkeypatch.setattr(publisher, "_run", lambda args, _cwd: subprocess.CompletedProcess(args, 0, "", ""))

    result = publish_pull_request(
        _settings(tmp_path),
        branch_name="retrieval-fix-retry",
        base_branch="multi-agent",
        validation_only=True,
    )

    assert result.status == "blocked"
    assert "requires a non-empty reason" in result.summary


def test_validation_only_publication_pushes_draft_without_production_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_base", lambda *_args: None)
    observed: dict[str, object] = {}

    def fake_run(args, _cwd):
        observed.setdefault("commands", []).append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    def fake_create(**kwargs):
        observed["request"] = kwargs
        return 42, "https://github.com/example/repo/pull/42"

    monkeypatch.setattr(publisher, "_run", fake_run)
    monkeypatch.setattr(publisher, "_github_repository", lambda _root: "example/repo")
    monkeypatch.setattr(publisher, "_create_draft_pr", fake_create)

    result = publish_pull_request(
        _settings(tmp_path),
        branch_name="retrieval-fix-retry",
        base_branch="multi-agent",
        validation_only=True,
        reason="synthetic defect fixture branch",
    )

    assert result.status == "published"
    assert result.validation_only is True
    assert result.pull_request_number == 42
    assert observed["request"]["title"].startswith("[VALIDATION ONLY]")
    assert "must not be merged" in observed["request"]["body"]
    assert "secret" not in result.model_dump_json()


def test_production_publication_rejects_nonancestor_branch_before_push(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(publisher, "_ensure_clean_base", lambda *_args: None)
    monkeypatch.setattr(publisher, "_load_production_artifact", lambda *_args: None)
    commands: list[list[str]] = []

    def fake_run(args, _cwd):
        commands.append(args)
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(publisher, "_run", fake_run)

    result = publish_pull_request(
        _settings(tmp_path),
        branch_name="retrieval-fix-retry",
        base_branch="multi-agent",
        artifact_dir=tmp_path,
    )

    assert result.status == "blocked"
    assert result.summary == "repair branch is not descended from base branch"
    assert not any(command[:2] == ["git", "push"] for command in commands)
