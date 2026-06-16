#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_peak(path: Path) -> dict[int, float]:
    peaks: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row["index"])
            used_mib = float(row["memory.used"])
            peaks[index] = max(peaks.get(index, 0.0), used_mib)
    return peaks


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize peak GPU memory from nvidia-smi CSV profiles.")
    parser.add_argument("--optimized", required=True, help="CSV from the optimized run.")
    parser.add_argument("--control", required=True, help="CSV from the control run.")
    args = parser.parse_args()

    optimized = read_peak(Path(args.optimized))
    control = read_peak(Path(args.control))
    gpu_ids = sorted(set(optimized) | set(control))

    print("gpu,control_peak_gb,optimized_peak_gb,saved_gb,saved_pct")
    total_control = 0.0
    total_optimized = 0.0
    for gpu_id in gpu_ids:
        control_gb = control.get(gpu_id, 0.0) / 1024
        optimized_gb = optimized.get(gpu_id, 0.0) / 1024
        saved_gb = control_gb - optimized_gb
        saved_pct = (saved_gb / control_gb * 100) if control_gb > 0 else 0.0
        total_control += control_gb
        total_optimized += optimized_gb
        print(f"{gpu_id},{control_gb:.2f},{optimized_gb:.2f},{saved_gb:.2f},{saved_pct:.1f}")

    if len(gpu_ids) > 1:
        saved = total_control - total_optimized
        saved_pct = (saved / total_control * 100) if total_control > 0 else 0.0
        print(f"total,{total_control:.2f},{total_optimized:.2f},{saved:.2f},{saved_pct:.1f}")


if __name__ == "__main__":
    main()
