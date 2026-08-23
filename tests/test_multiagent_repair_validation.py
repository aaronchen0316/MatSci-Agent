from __future__ import annotations

import subprocess
from pathlib import Path

import multiagent.repair_validation as repair_validation
from multiagent.repair_validation import validate_repair_test_evidence
from multiagent.settings import MultiAgentSettings


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "matsci_agent").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "matsci_agent" / "module.py").write_text("VALUE = 1\n")
    (repo / "tests" / "test_module.py").write_text("def test_value():\n    assert 1 == 1\n")
    _run(["git", "init", "-b", "multi-agent"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "base"], repo)
    _run(["git", "checkout", "-b", "retrieval-fix-1"], repo)
    return repo


def _settings(repo: Path, tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(tool_root=repo, target_repo=repo, target_base_branch="multi-agent", worktree_root=tmp_path / "worktrees")


def test_repair_validation_rejects_deleted_or_renamed_tests(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "src" / "matsci_agent" / "module.py").write_text("VALUE = 2\n")
    (repo / "tests" / "test_module.py").unlink()
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "unsafe test removal"], repo)

    evidence = validate_repair_test_evidence(
        _settings(repo, tmp_path),
        repo,
        declared_test_files=[],
        declared_test_targets=[],
    )

    assert evidence.status == "fail"
    assert evidence.deleted_or_renamed_test_files == ["tests/test_module.py"]
    assert "repair deletes or renames tests" in evidence.issues


def test_repair_validation_rejects_malformed_or_duplicate_test_targets(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "src" / "matsci_agent" / "module.py").write_text("VALUE = 2\n")
    (repo / "tests" / "test_module.py").write_text("def test_value():\n    assert VALUE == 2\n")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "repair with duplicate targets"], repo)

    evidence = validate_repair_test_evidence(
        _settings(repo, tmp_path),
        repo,
        declared_test_files=["tests/test_module.py"],
        declared_test_targets=["tests/test_module.py", "tests/test_module.py"],
    )

    assert evidence.status == "fail"
    assert "debugger test_targets contain malformed or duplicate paths" in evidence.issues


def test_repair_validation_rejects_per_file_coverage_regression(monkeypatch, tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "src" / "matsci_agent" / "module.py").write_text("VALUE = 2\n")
    (repo / "tests" / "test_module.py").write_text("def test_value():\n    assert True\n")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "coverage regression"], repo)

    real_run = repair_validation._run
    pytest_commands: list[list[str]] = []

    def fake_run(args, cwd):
        if args[0] == "git":
            return real_run(args, cwd)
        return subprocess.CompletedProcess(args, 0, "1 passed", "")

    def fake_pytest(args, _cwd, _environment_path):
        pytest_commands.append(args)
        return subprocess.CompletedProcess(args, 0, "1 passed", "")

    monkeypatch.setattr(repair_validation, "_run", fake_run)
    monkeypatch.setattr(repair_validation, "_run_pytest", fake_pytest)
    monkeypatch.setattr(
        repair_validation,
        "_baseline_coverage",
        lambda *_args: ({"src/matsci_agent/module.py": 90.0}, "baseline", None),
    )
    monkeypatch.setattr(
        repair_validation,
        "_coverage",
        lambda *_args: ({"src/matsci_agent/module.py": 80.0}, "repair"),
    )

    evidence = validate_repair_test_evidence(
        _settings(repo, tmp_path),
        repo,
        declared_test_files=["tests/test_module.py"],
        declared_test_targets=["tests/test_module.py"],
    )

    assert evidence.status == "fail"
    assert evidence.coverage_regressions == ["src/matsci_agent/module.py: 90.00% -> 80.00%"]
    assert "changed production-file coverage decreased" in evidence.issues
    assert any(command[:5] == ["uv", "run", "--extra", "dev", "pytest"] for command in pytest_commands)
