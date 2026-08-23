from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def aggregate(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=float)
    interval = None
    if len(array) > 1:
        interval = float(1.96 * np.std(array) / np.sqrt(len(array)))
    return {"mean": float(np.mean(array)), "ci95": interval}


def format_metric(value: Dict[str, Optional[float]]) -> str:
    if value["ci95"] is None:
        return "{:.4f}".format(value["mean"])
    return "{:.4f} +/- {:.4f}".format(value["mean"], value["ci95"])


def build_summary(
    raw: Dict[str, object],
    cells: Sequence[Tuple[str, str, str, float]],
    identity_hash: str,
) -> Dict[str, object]:
    rows = []
    for key, method, subset, alpha in cells:
        replications = raw["cells"].get(key, [])
        if len(replications) != raw["repetitions"]:
            raise RuntimeError("Cell {} is incomplete".format(key))
        metrics = {
            name: aggregate([float(rep["metrics"][name]) for rep in replications])
            for name in ("FID", "Diversity", "R@1", "R@2", "R@3")
        }
        rows.append(
            {
                "cell": key,
                "method": method,
                "subset": subset,
                "alpha": alpha,
                "metrics": metrics,
            }
        )
    return {
        "format_version": 2,
        "protocol_identity_sha256": identity_hash,
        "repetitions": raw["repetitions"],
        "rows": rows,
    }


def summary_text(summary: Dict[str, object], protocol: Dict[str, object]) -> str:
    lines = [
        "SafeMo MMU evaluation",
        "scope={} seed={} repetitions={} unsafe_ids_sha256={}".format(
            protocol["identity"]["dataset_scope"],
            protocol["identity"]["seed"],
            protocol["identity"]["repetitions"],
            protocol["identity"]["unsafe_ids_sha256"],
        ),
        "protocol=legacy-table2 effective_CFG=1.0 batch_size=32",
        "",
        "Method\tSubset\tAlpha\tFID\tDiversity\tR@1\tR@2\tR@3",
    ]
    for row in summary["rows"]:
        metrics = row["metrics"]
        lines.append(
            "{}\t{}\t{:g}\t{}\t{}\t{}\t{}\t{}".format(
                row["method"],
                row["subset"],
                row["alpha"],
                format_metric(metrics["FID"]),
                format_metric(metrics["Diversity"]),
                format_metric(metrics["R@1"]),
                format_metric(metrics["R@2"]),
                format_metric(metrics["R@3"]),
            )
        )
    return "\n".join(lines) + "\n"
