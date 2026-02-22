from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

try:
    import mlflow
except Exception:  # pragma: no cover - optional runtime dependency fallback
    mlflow = None


class MLflowLogger:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.enabled = mlflow is not None
        if self.enabled:
            mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, run_name: str):
        if not self.enabled:
            yield
            return
        with mlflow.start_run(run_name=run_name):
            yield

    def log_step(
        self,
        step_name: str,
        metrics: Mapping[str, float] | None = None,
        params: Mapping[str, str | int | float | bool] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if params:
            mlflow.log_params({f"{step_name}.{k}": v for k, v in params.items()})
        if metrics:
            mlflow.log_metrics({f"{step_name}.{k}": float(v) for k, v in metrics.items()})
