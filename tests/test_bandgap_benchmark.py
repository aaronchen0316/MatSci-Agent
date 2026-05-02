import json

from matsci_agent.evaluation.bandgap_benchmark import (
    _mae,
    _rmse,
    _spearman,
    write_bandgap_benchmark_artifacts,
)
from matsci_agent.schemas import BandGapBenchmarkArtifact, BandGapBenchmarkMetrics


def test_benchmark_metric_helpers():
    actual = [1.0, 2.0, 3.0]
    predicted = [1.5, 2.5, 3.5]

    assert _mae(actual, predicted) == 0.5
    assert round(_rmse(actual, predicted), 6) == 0.5
    assert round(_spearman(actual, predicted), 6) == 1.0


def test_benchmark_artifact_writer(tmp_path):
    artifact = BandGapBenchmarkArtifact(
        mode="small",
        metrics=BandGapBenchmarkMetrics(sample_size=1, mae=0.2, rmse=0.2, rank_correlation=None),
        rows=[
            {
                "material_id": "mp-1",
                "formula": "AlN",
                "actual_band_gap_ev": 5.7,
                "predicted_band_gap_ev": 5.5,
                "backend": "matgl",
                "band_gap_source": "matgl",
                "matgl_skipped_reason": None,
            }
        ],
    )

    output = tmp_path / "benchmark.json"
    write_bandgap_benchmark_artifacts(artifact, output)

    assert output.exists()
    assert output.with_suffix(".csv").exists()
    assert output.with_suffix(".md").exists()
    assert json.loads(output.read_text())["metrics"]["mae"] == 0.2
