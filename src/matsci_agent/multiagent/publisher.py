from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from matsci_agent.multiagent.schemas import HarnessRunReport, PullRequestPublication
from matsci_agent.multiagent.settings import MultiAgentSettings

_BRANCH_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def _validate_branch(branch: str) -> str:
    if not _BRANCH_PATTERN.fullmatch(branch) or ".." in branch or branch.endswith((".lock", ".")):
        raise ValueError("branch must be a safe single path segment")
    return branch


def _github_repository(repo_root: Path) -> str:
    remote = _output(_run(["git", "remote", "get-url", "origin"], repo_root))
    match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/\s]+)/([^/\s]+?)(?:\.git)?", remote)
    if not match:
        raise ValueError("origin must be a GitHub repository remote")
    return f"{match.group(1)}/{match.group(2)}"


def _ensure_clean_base(repo_root: Path, base_branch: str) -> str | None:
    if _output(_run(["git", "status", "--porcelain"], repo_root)):
        return "current checkout must be clean before publication"
    base = _run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{base_branch}"], repo_root)
    return None if base.returncode == 0 else f"base branch does not exist: {base_branch}"


def _run_branch_suite(settings: MultiAgentSettings, branch_name: str) -> tuple[bool, str]:
    root = settings.worktree_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    worktree = root / f".publish-ci-{uuid4().hex[:12]}"
    add = _run(["git", "worktree", "add", "--detach", str(worktree), branch_name], settings.repo_root)
    if add.returncode != 0:
        return False, f"unable to create CI worktree: {_output(add)}"
    try:
        result = _run(["uv", "run", "--extra", "dev", "pytest", "-q"], worktree)
        return result.returncode == 0, _output(result)
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)], settings.repo_root)


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
    if report.latest_verifier_report is None or report.latest_verifier_report.status != "accepted":
        return "harness artifact lacks accepted verifier evidence"
    if report.latest_repair_test_evidence is None or report.latest_repair_test_evidence.status != "pass":
        return "harness artifact lacks passing deterministic repair-test evidence"
    live = report.latest_tester_report.live_evaluation if report.latest_tester_report else None
    if live is None or live.status != "pass" or not live.real_source_used:
        return "harness artifact lacks passing live Materials Project evidence"
    return None


def _create_draft_pr(
    *,
    repository: str,
    token: str,
    branch_name: str,
    base_branch: str,
    title: str,
    body: str,
) -> tuple[int, str]:
    payload = json.dumps(
        {
            "title": title,
            "head": branch_name,
            "base": base_branch,
            "body": body,
            "draft": True,
        }
    ).encode()
    request = Request(
        f"https://api.github.com/repos/{repository}/pulls",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode())
    except HTTPError as exc:
        raise RuntimeError(f"GitHub PR creation failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("GitHub PR creation request failed") from exc
    return int(result["number"]), str(result["html_url"])


def publish_pull_request(
    settings: MultiAgentSettings,
    *,
    branch_name: str,
    base_branch: str,
    artifact_dir: Path | None = None,
    validation_only: bool = False,
    reason: str | None = None,
) -> PullRequestPublication:
    """Push one validated repair branch and create a draft GitHub pull request."""

    try:
        branch_name = _validate_branch(branch_name)
        base_branch = _validate_branch(base_branch)
    except ValueError as exc:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary=str(exc))
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary="missing GITHUB_TOKEN")
    clean_error = _ensure_clean_base(settings.repo_root, base_branch)
    if clean_error:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary=clean_error)
    branch = _run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], settings.repo_root)
    if branch.returncode != 0:
        return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary="repair branch does not exist")

    if validation_only:
        if not reason or not reason.strip():
            return PullRequestPublication(
                status="blocked",
                branch_name=branch_name,
                base_branch=base_branch,
                validation_only=True,
                summary="validation-only publication requires a non-empty reason",
            )
        title = f"[VALIDATION ONLY] {branch_name} -> {base_branch}"
        body = "\n".join(
            [
                "This draft is validation-only and must not be merged.",
                "It bypassed production repair gates for inspection of an unsafe or non-ancestor branch.",
                f"Reason: {reason.strip()}",
            ]
        )
        ci_output = ""
    else:
        if artifact_dir is None:
            return PullRequestPublication(status="blocked", branch_name=branch_name, base_branch=base_branch, summary="production publication requires --artifact-dir")
        artifact_error = _load_production_artifact(artifact_dir.resolve(), branch_name)
        if artifact_error:
            return PullRequestPublication(
                status="blocked",
                branch_name=branch_name,
                base_branch=base_branch,
                artifact_dir=str(artifact_dir),
                summary=artifact_error,
            )
        ancestor = _run(["git", "merge-base", "--is-ancestor", base_branch, branch_name], settings.repo_root)
        if ancestor.returncode != 0:
            return PullRequestPublication(
                status="blocked",
                branch_name=branch_name,
                base_branch=base_branch,
                artifact_dir=str(artifact_dir),
                summary="repair branch is not descended from base branch",
            )
        diff_check = _run(["git", "diff", "--check", f"{base_branch}...{branch_name}"], settings.repo_root)
        if diff_check.returncode != 0:
            return PullRequestPublication(
                status="blocked",
                branch_name=branch_name,
                base_branch=base_branch,
                artifact_dir=str(artifact_dir),
                summary=f"repair diff check failed: {_output(diff_check)}",
            )
        passed, ci_output = _run_branch_suite(settings, branch_name)
        if not passed:
            return PullRequestPublication(
                status="blocked",
                branch_name=branch_name,
                base_branch=base_branch,
                artifact_dir=str(artifact_dir),
                local_ci_output=ci_output,
                summary="local CI suite failed",
            )
        title = f"Repair live scenario on {branch_name}"
        body = "\n".join(
            [
                "Automated repair candidate. Draft until human review and GitHub Actions pass.",
                f"Validated artifact: `{artifact_dir.resolve()}`",
            ]
        )

    push = _run(["git", "push", "origin", f"refs/heads/{branch_name}:refs/heads/{branch_name}"], settings.repo_root)
    if push.returncode != 0:
        return PullRequestPublication(
            status="failed",
            branch_name=branch_name,
            base_branch=base_branch,
            validation_only=validation_only,
            artifact_dir=str(artifact_dir) if artifact_dir else None,
            summary=f"git push failed: {_output(push)}",
        )
    try:
        number, url = _create_draft_pr(
            repository=_github_repository(settings.repo_root),
            token=token,
            branch_name=branch_name,
            base_branch=base_branch,
            title=title,
            body=body,
        )
    except (ValueError, RuntimeError) as exc:
        return PullRequestPublication(
            status="failed",
            branch_name=branch_name,
            base_branch=base_branch,
            validation_only=validation_only,
            artifact_dir=str(artifact_dir) if artifact_dir else None,
            summary=str(exc),
        )
    return PullRequestPublication(
        status="published",
        branch_name=branch_name,
        base_branch=base_branch,
        validation_only=validation_only,
        artifact_dir=str(artifact_dir) if artifact_dir else None,
        local_ci_output=ci_output,
        summary="draft pull request created",
        pull_request_number=number,
        pull_request_url=url,
    )
