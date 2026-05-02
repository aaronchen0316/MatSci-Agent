from __future__ import annotations

import argparse

from matsci_agent.evaluation.bandgap_benchmark import run_bandgap_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument(
        "--output",
        default="artifacts/bandgap_benchmark.json",
        help="Path for JSON artifact output.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    artifact = run_bandgap_benchmark(
        mode=args.mode,
        output_path=args.output,
        sample_size=args.sample_size,
    )
    print(artifact.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
