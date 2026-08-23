from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from multiagent.schemas import HarnessRunReport, PullRequestPublication
from multiagent.settings import MultiAgentSettings

_BRANCH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SUCCESSFUL_CHECKS = {"success", "neutral", "skipped"}


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def _validate_branch(branch: str, *, require_fix: bool = False) -> str:
    parts = branch.split("/")
    if not parts or any(
        not _BRANCH_SEGMENT.fullmatch(part) or ".." in part or part.endswith((".", ".lock"))
        for part in parts
    ):
        raise ValueError("branch must contain safe Git path segments")
    if require_fix and (len(parts) != 2 or parts[0] != "fix"):
        raise ValueError("repair branch must use fix/<issue>")
    return branch


def _github_repository(target_repo: Path) -> str:
    remote = _output(_run(["git", "remote", "get-url", "origin"], target_repo))
    match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/\s]+)/([^/\s]+?)(?:\.git)?", remote)
    if not match:
        raise ValueError("origin must be a GitHub repository remote")
    return f"{match.group(1)}/{match.group(2)}"


def _ensure_clean_tooling(settings: MultiAgentSettings) -> str | None:
    if _output(_run(["git", "status", "--porcelain"], settings.resolved_tool_root)):
        return "tooling checkout must be clean before publication"
    base = _run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{settings.target_base_branch}"],
        settings.resolved_target_repo,
    )
    return None if base.returncode == 0 else f"target base branch does not exist: {settings.target_base_branch}"


def _run_branch_suite(settings: MultiAgentSettings, branch_name: str) -> tuple[bool, str]:
    root = settings.worktree_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    worktree = root / f".publish-ci-{uuid4().hex[:12]}"
    add = _run(["git", "worktree", "add", "--detach", str(worktree), branch_name], settings.resolved_target_repo)
    if add.returncode != 0:
        return False, f"unable to create CI worktree: {_output(add)}"
    try:
        result = _run(["uv", "run", "--extra", "dev", "pytest", "-q"], worktree)
        return result.returncode == 0, _output(result)
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)], settings.resolved_target_repo)


def _load_production_artifact(artifact_dir: Path, branch_name: str) -> str | None:
    report_path = artifact_dir / "harness_run_report.json"
    if not report_path.is_file():
        return "artifact directory does not contain harness_run_report.json"
    try:
        report = HarnessRunReport.model_validate_json(report_path.read_text())
    except Exception:
        return "harness run artifact is malformed"
    if report.status != "pass" or report.stop_reason != "dual_review_pass":
        return "harness artifact is not a successful dual-review repair"
    if report.branch_name != branch_name:
        return "harness artifact branch does not match requested branch"
    if report.latest_debugger_report is None or report.latest_debugger_report.status != "patched":
        return "harness artifact lacks a committed debugger repair"
    if report.latest_verifier_report is None or report.latest_verifier_report.status != "accepted":
        return "harness artifact lacks accepted verifier evidence"
    if report.latest_repair_test_evidence is None or report.latest_repair_test_evidence.status != "pass":
        return "harness artifact lacks passing deterministic repair-test evidence"
    live = report.latest_tester_report.live_evaluation if report.latest_tester_report else None
    if live is None or live.status != "pass" or not live.real_source_used:
        return "harness artifact lacks passing live Materials Project evidence"
    return None


def _product_only_diff(settings: MultiAgentSettings, branch_name: str) -> str | None:
    result = _run(
        ["git", "diff", "--name-only", f"{settings.target_base_branch}...{branch_name}"],
        settings.resolved_target_repo,
    )
    if result.returncode != 0:
        return f"unable to inspect repair diff: {_output(result)}"
    paths = [path for path in result.stdout.splitlines() if path]
    disallowed = [path for path in paths if not (path.startswith("src/matsci_agent/") or path.startswith("tests/"))]
    if disallowed:
        return f"repair diff includes non-product paths: {', '.join(disallowed)}"
    return None


def _github_json(
    *,
    method: str,
    url: str,
    token: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except HTTPError as exc:
        raise RuntimeError(f"GitHub API request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("GitHub API request failed") from exc


def _create_ready_pr(
    *, repository: str, token: str, branch_name: str, base_branch: str, artifact_dir: Path
) -> tuple[int, str, str]:
    result = _github_json(
        method="POST",
        url=f"https://api.github.com/repos/{repository}/pulls",
        token=token,
        payload={
            "title": f"Automated repair: {branch_name}",
            "head": branch_name,
            "base": base_branch,
            "body": "\n".join(
                [
                    "Automated MatSci-Agent repair.",
                    "Passed debugger tests, full local suite, coverage comparison, verifier review, and fresh live retest.",
                    f"Local evidence: `{artifact_dir}`",
                ]
            ),
            "draft": False,
        },
    )
    return int(result["number"]), str(result["html_url"]), str(result["head"]["sha"])


def _wait_for_checks(*, repository: str, token: str, sha: str, timeout_seconds: int = 900) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _github_json(
            method="GET",
            url=f"https://api.github.com/repos/{repository}/commits/{sha}/check-runs",
            token=token,
        )
        checks = list(result.get("check_runs") or [])
        if checks and all(check.get("status") == "completed" for check in checks):
            conclusions = {str(check.get("conclusion")) for check in checks}
            return "pass" if conclusions.issubset(_SUCCESSFUL_CHECKS) else "fail"
        time.sleep(10)
    return "timeout"


def _squash_merge(*, repository: str, token: str, number: int, sha: str) -> str | None:
    result = _github_json(
        method="PUT",
        url=f"https://api.github.com/repos/{repository}/pulls/{number}/merge",
        token=token,
        payload={"merge_method": "squash", "sha": sha},
    )
    return str(result.get("sha")) if result.get("merged") else None


def publish_and_merge_repair(
    settings: MultiAgentSettings,
    *,
    branch_name: str,
    artifact_dir: Path,
) -> PullRequestPublication:
    """Publish one fully validated product repair and squash-merge after CI."""

    base_branch = settings.target_base_branch
    try:
        branch_name = _validate_branch(branch_name, require_fix=True)
        _validate_branch(base_branch)
    except ValueError as exc:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary=str(exc))
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary="missing GITHUB_TOKEN")
    clean_error = _ensure_clean_tooling(settings)
    if clean_error:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary=clean_error)
    branch = _run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], settings.resolved_target_repo)
    if branch.returncode != 0:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary="repair branch does not exist")
    artifact_error = _load_production_artifact(artifact_dir.resolve(), branch_name)
    if artifact_error:
        return PullRequestPublication(
            status="blocked", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), summary=artifact_error
        )
    ancestor = _run(["git", "merge-base", "--is-ancestor", base_branch, branch_name], settings.resolved_target_repo)
    if ancestor.returncode != 0:
        return PullRequestPublication(
            status="blocked", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), summary="repair branch is not descended from target main"
        )
    product_error = _product_only_diff(settings, branch_name)
    if product_error:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), summary=product_error)
    diff_check = _run(["git", "diff", "--check", f"{base_branch}...{branch_name}"], settings.resolved_target_repo)
    if diff_check.returncode != 0:
        return PullRequestPublication(
            status="blocked", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), summary=f"repair diff check failed: {_output(diff_check)}"
        )
    passed, ci_output = _run_branch_suite(settings, branch_name)
    if not passed:
        return PullRequestPublication(
            status="blocked", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), local_ci_output=ci_output, summary="local CI suite failed"
        )
    push = _run(["git", "push", "origin", f"refs/heads/{branch_name}:refs/heads/{branch_name}"], settings.resolved_target_repo)
    if push.returncode != 0:
        return PullRequestPublication(
            status="failed", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), local_ci_output=ci_output, summary=f"git push failed: {_output(push)}"
        )
    try:
        repository = _github_repository(settings.resolved_target_repo)
        number, url, sha = _create_ready_pr(
            repository=repository,
            token=token,
            branch_name=branch_name,
            base_branch=base_branch,
            artifact_dir=artifact_dir.resolve(),
        )
        ci_status = _wait_for_checks(repository=repository, token=token, sha=sha)
        if ci_status != "pass":
            return PullRequestPublication(
                status="published",
                branch_name=branch_name,
                base_branch=base_branch,
                artifact_dir=str(artifact_dir),
                local_ci_output=ci_output,
                summary=f"ready pull request created; remote CI {ci_status}, not merged",
                pull_request_number=number,
                pull_request_url=url,
                head_sha=sha,
                ci_status=ci_status,
            )
        merge_sha = _squash_merge(repository=repository, token=token, number=number, sha=sha)
    except (ValueError, RuntimeError) as exc:
        return PullRequestPublication(
            status="failed", branch_name=branch_name, base_branch=base_branch, artifact_dir=str(artifact_dir), local_ci_output=ci_output, summary=str(exc)
        )
    if not merge_sha:
        return PullRequestPublication(
            status="published",
            branch_name=branch_name,
            base_branch=base_branch,
            artifact_dir=str(artifact_dir),
            local_ci_output=ci_output,
            summary="ready pull request created; GitHub rejected squash merge",
            pull_request_number=number,
            pull_request_url=url,
            head_sha=sha,
            ci_status="pass",
        )
    return PullRequestPublication(
        status="merged",
        branch_name=branch_name,
        base_branch=base_branch,
        artifact_dir=str(artifact_dir),
        local_ci_output=ci_output,
        summary="remote CI passed; squash merge completed",
        pull_request_number=number,
        pull_request_url=url,
        head_sha=sha,
        merge_sha=merge_sha,
        ci_status="pass",
    )
