from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from matsci_agent.multiagent.settings import MultiAgentSettings


class HarnessArtifactStore:
    """Write non-secret evidence for one harness or live-evaluation run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    @classmethod
    def create(cls, settings: MultiAgentSettings, objective: str) -> "HarnessArtifactStore":
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        run_dir = settings.resolved_artifact_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        store = cls(run_dir)
        store.write_json(
            "manifest.json",
            {
                "objective": objective,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": settings.model,
                "disable_tracing": settings.disable_tracing,
                "enable_live_mp": settings.enable_live_mp,
                "enable_git_write": settings.enable_git_write,
                "base_branch": settings.base_branch,
            },
        )
        return store

    def write_model(self, relative_path: str, payload: BaseModel) -> None:
        self.write_json(relative_path, payload.model_dump(mode="json"))

    def write_json(self, relative_path: str, payload: object) -> None:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    def write_text(self, relative_path: str, content: str) -> None:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _path(self, relative_path: str) -> Path:
        path = (self.run_dir / relative_path).resolve()
        if self.run_dir not in path.parents:
            raise ValueError(f"artifact path escapes run directory: {relative_path}")
        return path
