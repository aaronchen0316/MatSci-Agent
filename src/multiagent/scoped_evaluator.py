from __future__ import annotations

import sys

from multiagent.evaluator import LiveRetrievalEvaluator
from multiagent.schemas import LiveEvalEvidence, LiveEvalInput


def main() -> int:
    """Evaluate one payload from stdin using this process's source checkout."""

    raw = sys.stdin.read()
    query = ""
    try:
        payload = LiveEvalInput.model_validate_json(raw)
        query = payload.query
        evidence = LiveRetrievalEvaluator().evaluate(payload)
    except Exception as exc:
        evidence = LiveEvalEvidence(
            status="blocked",
            query=query,
            blocked_reason=f"scoped evaluator failed: {type(exc).__name__}",
        )
    sys.stdout.write(evidence.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
