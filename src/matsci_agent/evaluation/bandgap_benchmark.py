from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

from matsci_agent.config import settings
from matsci_agent.schemas import (
    BandGapBenchmarkArtifact,
    BandGapBenchmarkMetrics,
    Candidate,
    PropertyPredictorInput,
)
from matsci_agent.tools.property_predictor import PropertyPredictor


def run_bandgap_benchmark(
    mode: str,
    output_path: str | Path,
    sample_size: int | None = None,
) -> BandGapBenchmarkArtifact:
    sample_count = sample_size or (10 if mode == "small" else 50)
    candidates = load_mp_bandgap_benchmark_candidates(limit=sample_count)

    predictor = PropertyPredictor()
    result = predictor.run(
        PropertyPredictorInput(
            candidates=candidates,
            goal="benchmark band gap predictor",
            calculate_matgl=True,
            recalculate_top_n=min(len(candidates), sample_count),
            matgl_max_recalc_entries=sample_count,
            matgl_max_atoms=settings.matgl_max_atoms,
            enable_relaxation=False,
            relaxation_max_steps=settings.matgl_relaxation_max_steps,
        )
    )

    rows: list[dict[str, Any]] = []
    predicted_values: list[float] = []
    actual_values: list[float] = []
    failure_count = 0
    fallback_count = 0
    mp_passthrough_count = 0
    matgl_count = 0
    skipped_count = 0

    for prediction in result.predictions:
        actual = prediction.candidate.features.get("mp_band_gap_ev")
        backend = prediction.predicted.backend
        skipped = prediction.candidate.features.get("matgl_skipped_reason")
        band_gap_source = prediction.candidate.features.get("band_gap_source")
        if skipped is not None:
            skipped_count += 1
        if backend.startswith("m3gnet_structure_fallback:"):
            fallback_count += 1
        elif band_gap_source == "materials_project":
            mp_passthrough_count += 1
        else:
            matgl_count += 1

        if actual is None:
            failure_count += 1
            continue

        predicted = float(prediction.predicted.band_gap_ev)
        actual_float = float(actual)
        predicted_values.append(predicted)
        actual_values.append(actual_float)
        rows.append(
            {
                "material_id": prediction.candidate.material_id,
                "formula": prediction.candidate.formula,
                "actual_band_gap_ev": actual_float,
                "predicted_band_gap_ev": predicted,
                "backend": backend,
                "band_gap_source": band_gap_source,
                "matgl_skipped_reason": skipped,
            }
        )

    metrics = BandGapBenchmarkMetrics(
        sample_size=len(rows),
        mae=_mae(actual_values, predicted_values),
        rmse=_rmse(actual_values, predicted_values),
        rank_correlation=_spearman(actual_values, predicted_values),
        failure_count=failure_count,
        skipped_count=skipped_count,
        mp_passthrough_count=mp_passthrough_count,
        matgl_count=matgl_count,
        fallback_count=fallback_count,
    )
    artifact = BandGapBenchmarkArtifact(
        mode="small" if mode == "small" else "full",
        metrics=metrics,
        rows=rows,
    )
    write_bandgap_benchmark_artifacts(artifact, output_path)
    return artifact


def load_mp_bandgap_benchmark_candidates(limit: int) -> list[Candidate]:
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY required for benchmark sampling.")

    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError(f"mp-api import failed: {exc}") from exc

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(0.01, 12.0),
            energy_above_hull=(0.0, 0.2),
            fields=[
                "material_id",
                "formula_pretty",
                "elements",
                "band_gap",
                "energy_above_hull",
                "nsites",
                "structure",
            ],
            is_metal=False,
            num_elements=(2, 6),
            num_chunks=1,
            chunk_size=max(120, limit * 10),
        )

    candidates: list[Candidate] = []
    for doc in docs:
        nsites = getattr(doc, "nsites", None)
        structure = getattr(doc, "structure", None)
        band_gap = getattr(doc, "band_gap", None)
        if not isinstance(nsites, (int, float)) or int(nsites) >= settings.matgl_max_atoms:
            continue
        if structure is None or band_gap is None:
            continue
        candidates.append(
            Candidate(
                material_id=str(doc.material_id),
                formula=str(doc.formula_pretty),
                source="materials_project",
                features={
                    "elements": [str(e) for e in getattr(doc, "elements", [])],
                    "mp_band_gap_ev": float(band_gap),
                    "mp_energy_above_hull": getattr(doc, "energy_above_hull", None),
                    "nsites": nsites,
                    "structure": structure.as_dict(),
                },
            )
        )
        if len(candidates) >= limit:
            break

    if not candidates:
        raise RuntimeError("No benchmark candidates retrieved from Materials Project.")
    return candidates


def write_bandgap_benchmark_artifacts(
    artifact: BandGapBenchmarkArtifact,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2))

    csv_path = path.with_suffix(".csv")
    if artifact.rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(artifact.rows[0].keys()))
            writer.writeheader()
            writer.writerows(artifact.rows)

    md_path = path.with_suffix(".md")
    md_path.write_text(_markdown_summary(artifact))


def _mae(actual: list[float], predicted: list[float]) -> float | None:
    if not actual or len(actual) != len(predicted):
        return None
    return mean(abs(a - p) for a, p in zip(actual, predicted))


def _rmse(actual: list[float], predicted: list[float]) -> float | None:
    if not actual or len(actual) != len(predicted):
        return None
    return math.sqrt(mean((a - p) ** 2 for a, p in zip(actual, predicted)))


def _spearman(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2 or len(actual) != len(predicted):
        return None
    ranked_actual = _rank(actual)
    ranked_predicted = _rank(predicted)
    return _pearson(ranked_actual, ranked_predicted)


def _rank(values: list[float]) -> list[float]:
    sorted_pairs = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _markdown_summary(artifact: BandGapBenchmarkArtifact) -> str:
    metrics = artifact.metrics
    lines = [
        "# Band-gap Benchmark",
        "",
        f"- mode: `{artifact.mode}`",
        f"- sample_size: `{metrics.sample_size}`",
        f"- mae: `{metrics.mae}`",
        f"- rmse: `{metrics.rmse}`",
        f"- rank_correlation: `{metrics.rank_correlation}`",
        f"- failure_count: `{metrics.failure_count}`",
        f"- skipped_count: `{metrics.skipped_count}`",
        f"- mp_passthrough_count: `{metrics.mp_passthrough_count}`",
        f"- matgl_count: `{metrics.matgl_count}`",
        f"- fallback_count: `{metrics.fallback_count}`",
    ]
    return "\n".join(lines) + "\n"
