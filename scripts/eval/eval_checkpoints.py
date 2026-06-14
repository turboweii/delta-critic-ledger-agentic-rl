#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "checkpoints_delta_grpo.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = load_config(ROOT / cfg["base_config"])
    out_root = ensure_dir(ROOT / "outputs" / cfg["experiment_name"])
    manifest = []

    for ckpt in cfg["checkpoints"]:
        run_cfg = json.loads(json.dumps(base))
        run_cfg["policy"]["model_name"] = ckpt["model_name"]
        run_cfg["policy"]["served_model_name"] = ckpt["served_model_name"]
        run_cfg["output"]["dir"] = str(out_root / f"step_{ckpt['step']}")
        cfg_path = out_root / f"eval_step_{ckpt['step']}.json"
        cfg_path.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        cmd = ["python3", "scripts/eval/run_policy_eval.py", "--config", str(cfg_path)]
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, cwd=ROOT, check=True)
        manifest.append({"step": ckpt["step"], "config": str(cfg_path), "output_dir": run_cfg["output"]["dir"]})

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote checkpoint eval manifest to {out_root}")


if __name__ == "__main__":
    main()

