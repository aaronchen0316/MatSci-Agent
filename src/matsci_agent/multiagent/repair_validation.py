from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from matsci_agent.multiagent.schemas import RepairTestEvidence
from matsci_agent.multiagent.settings import MultiAgentSettings


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def _relative_test_path(value: str) -> str | None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        return None
    normalized = path.as_posix()
    return normalized if normalized.startswith("tests/") else None


def _changed_paths(repo_root: Path, base_branch: str) -> tuple[list[str], list[str], list[str]]:
    result = _run(["git", "diff", "--name-status", "-M", f"{base_branch}...HEAD"], repo_root)
    if result.returncode != 0:
        raise ValueError(f"unable to diff repair branch against {base_branch}: {_output(result)}")

    sources: list[str] = []
    tests: list[str] = []
    removed_tests: list[str] = []
    for line in (result.stdout or "").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0]
        paths = fields[1:]
        if status.startswith(("D", "R")):
            for path in paths:
                if _relative_test_path(path):
                    removed_tests.append(path)
        final_path = paths[-1]
        if final_path.startswith("src/matsci_agent/") and final_path.endswith(".py") and not status.startswith("D"):
            sources.append(final_path)
        if _relative_test_path(final_path) and not status.startswith("D"):
            tests.append(final_path)
    return sorted(set(sources)), sorted(set(tests)), sorted(set(removed_tests))


def _coverage(repo_root: Path, output_path: Path) -> tuple[dict[str, float], str]:
    result = _run(
        [
            "uv",
            "run",
            "pytest",
            "-q",
            "--cov=src/matsci_agent",
            f"--cov-report=json:{output_path}",
        ],
        repo_root,
    )
    output = _output(result)
    if result.returncode != 0 or not output_path.is_file():
        return {}, output
    try:
        payload = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, output

    values: dict[str, float] = {}
    for raw_path, details in payload.get("files", {}).items():
        path = Path(raw_path)
        try:
            relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative = path.as_posix()
        summary = details.get("summary", {})
        values[relative] = float(summary.get("percent_covered", 0.0))
    return values, output


def _baseline_coverage(settings: MultiAgentSettings, base_branch: str, output_path: Path) -> tuple[dict[str, float], str, str | None]:
    root = settings.worktree_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = root / f".coverage-base-{uuid4().hex[:12]}"
    add = _run(["git", "worktree", "add", "--detach", str(baseline), base_branch], settings.repo_root)
    if add.returncode != 0:
        return {}, "", f"unable to create baseline worktree: {_output(add)}"
    try:
        coverage, output = _coverage(baseline, output_path)
        if not coverage:
            return {}, output, "baseline coverage command failed"
        return coverage, output, None
    finally:
        _run(["git", "worktree", "remove", "--force", str(baseline)], settings.repo_root)


def validate_repair_test_evidence(
    settings: MultiAgentSettings,
    worktree_path: Path,
    *,
    declared_test_files: list[str],
    declared_test_targets: list[str],
) -> RepairTestEvidence:
    """Run deterministic test and coverage gates for a committed repair worktree."""

    repo_root = worktree_path.resolve()
    issues: list[str] = []
    try:
        changed_sources, changed_tests, removed_tests = _changed_paths(repo_root, settings.base_branch)
    except ValueError as exc:
        return RepairTestEvidence(status="blocked", issues=[str(exc)])

    normalized_declared_files = sorted({path for value in declared_test_files if (path := _relative_test_path(value))})
    normalized_targets = sorted({path for value in declared_test_targets if (path := _relative_test_path(value))})
    if not changed_sources:
        issues.append("repair did not change a production Python file")
    if not changed_tests:
        issues.append("repair did not add or modify a Python test")
    if removed_tests:
        issues.append("repair deletes or renames tests")
    if normalized_declared_files != changed_tests:
        issues.append("debugger test_files must exactly match changed test files")
    if not normalized_targets or not set(changed_tests).issubset(normalized_targets):
        issues.append("debugger test_targets must execute every changed test file")
    if len(normalized_targets) != len(declared_test_targets):
        issues.append("debugger test_targets contain malformed or duplicate paths")

    targeted_output = ""
    full_output = ""
    if not issues:
        collect = _run(["uv", "run", "pytest", "--collect-only", "-q", *normalized_targets], repo_root)
        targeted = _run(["uv", "run", "pytest", "-q", *normalized_targets], repo_root)
        targeted_output = _output(collect) + "\n" + _output(targeted)
        if collect.returncode != 0:
            issues.append("changed tests do not collect")
        if targeted.returncode != 0:
            issues.append("changed tests do not pass")

    coverage_before: dict[str, float] = {}
    coverage_after: dict[str, float] = {}
    regressions: list[str] = []
    with tempfile.TemporaryDirectory(prefix="matsci-repair-coverage-") as directory:
        temporary = Path(directory)
        if not issues:
            coverage_before, baseline_output, baseline_error = _baseline_coverage(
                settings,
                settings.base_branch,
                temporary / "baseline.json",
            )
            coverage_after, repair_coverage_output = _coverage(repo_root, temporary / "repair.json")
            full_output = "\n".join(item for item in [baseline_output, repair_coverage_output] if item)
            if baseline_error:
                issues.append(baseline_error)
            if not coverage_after:
                issues.append("repair coverage command failed")
            for source in changed_sources:
                before = coverage_before.get(source, 0.0)
                after = coverage_after.get(source, 0.0)
                if after + 1e-9 < before:
                    regressions.append(f"{source}: {before:.2f}% -> {after:.2f}%")
            if regressions:
                issues.append("changed production-file coverage decreased")

    return RepairTestEvidence(
        status="pass" if not issues else "fail",
        changed_source_files=changed_sources,
        changed_test_files=changed_tests,
        deleted_or_renamed_test_files=removed_tests,
        declared_test_files=declared_test_files,
        declared_test_targets=declared_test_targets,
        targeted_test_output=targeted_output,
        full_test_output=full_output,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        coverage_regressions=regressions,
        issues=issues,
    )
