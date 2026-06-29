from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return the score produced by TauBenchInteraction."""
    del solution_str, ground_truth, kwargs
    if data_source != "tau_bench_airline":
        raise ValueError(f"Unsupported data source: {data_source}")
    info = dict(extra_info or {})
    score = float(info.get("final_score", 0.0))
    diagnostics = dict(info.get("score_info") or {})
    diagnostics.pop("score", None)
    return {"score": score, **diagnostics}

