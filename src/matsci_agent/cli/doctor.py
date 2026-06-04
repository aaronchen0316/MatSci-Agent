from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from matsci_agent.cli.transport import probe_health
from matsci_agent.nlp.parser import resolve_llm_api_key_env


@dataclass(frozen=True)
class DoctorCheck:
    status: str
    name: str
    detail: str


def run_doctor_checks(api_url: str | None = None) -> list[DoctorCheck]:
    checks = [
        _check_python_version(),
        _check_package("matsci_agent", "Core package"),
        _check_package("fastapi", "FastAPI"),
        _check_package("typer", "Typer CLI"),
        _check_package("rich", "Rich renderer"),
        _check_package("mlflow", "MLflow"),
        _check_package("mp_api", "mp-api", optional=True),
        _check_package("matgl", "MatGL", optional=True),
        _check_package("torch", "PyTorch", optional=True),
        _check_package("dgl", "DGL", optional=True),
        _check_env("MP_API_KEY", "Materials Project live retrieval", optional=True),
        _check_env(resolve_llm_api_key_env(), "OpenRouter parsing / policy filter", optional=False),
        _check_path(
            Path(os.getenv("MATSCI_MATGL_MODEL", "models/pretrained/MEGNet-MP-2019.4.1-BandGap-mfi")),
            "MatGL band-gap model path",
            optional=True,
        ),
        _check_path(
            Path(os.getenv("MATSCI_MATGL_RELAX_MODEL", "models/pretrained/TensorNet-PES-MatPES-PBE-2025.2")),
            "MatGL relax model path",
            optional=True,
        ),
    ]
    if api_url:
        checks.append(_check_api_health(api_url))
    return checks


def has_failures(checks: list[DoctorCheck]) -> bool:
    return any(check.status == "FAIL" for check in checks)


def _check_python_version() -> DoctorCheck:
    version = sys.version_info
    if (version.major, version.minor) == (3, 12):
        return DoctorCheck("PASS", "Python runtime", f"{version.major}.{version.minor} supported.")
    if version.major == 3 and version.minor in {11, 12}:
        return DoctorCheck("WARN", "Python runtime", f"{version.major}.{version.minor} supported, 3.12 preferred for DGL-backed MatGL path.")
    return DoctorCheck("FAIL", "Python runtime", f"{version.major}.{version.minor} outside supported range >=3.11,<3.13.")


def _check_package(module_name: str, label: str, optional: bool = False) -> DoctorCheck:
    exists = importlib.util.find_spec(module_name) is not None
    if exists:
        return DoctorCheck("PASS", label, "Import available.")
    if optional:
        return DoctorCheck("WARN", label, "Optional package missing.")
    return DoctorCheck("FAIL", label, "Required package missing.")


def _check_env(name: str, label: str, optional: bool = False) -> DoctorCheck:
    if os.getenv(name):
        return DoctorCheck("PASS", label, f"{name} set.")
    if optional:
        return DoctorCheck("WARN", label, f"{name} missing.")
    return DoctorCheck("FAIL", label, f"{name} missing.")


def _check_path(path: Path, label: str, optional: bool = False) -> DoctorCheck:
    if path.exists():
        return DoctorCheck("PASS", label, str(path))
    if optional:
        return DoctorCheck("WARN", label, f"Missing path: {path}")
    return DoctorCheck("FAIL", label, f"Missing path: {path}")


def _check_api_health(api_url: str) -> DoctorCheck:
    ok, detail = probe_health(api_url)
    return DoctorCheck("PASS" if ok else "FAIL", "API health", detail)
