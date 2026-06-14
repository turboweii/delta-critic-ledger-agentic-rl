#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir, load_config


def wait_for_endpoint(base_url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"Timed out waiting for {url}")


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "checkpoints_delta_grpo.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assistant-gpus", default="0")
    parser.add_argument("--assistant-gpu-memory-utilization", default="0.78")
    parser.add_argument("--assistant-max-model-len", default="16384")
    parser.add_argument("--server-timeout", type=int, default=600)
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = load_config(ROOT / cfg["base_config"])
    out_root = ensure_dir(ROOT / "outputs" / cfg["experiment_name"])
    manifest = []

    if not args.dry_run:
        wait_for_endpoint(base["env"]["user_base_url"], None, 15)

    for ckpt in cfg["checkpoints"]:
        run_cfg = json.loads(json.dumps(base))
        run_cfg["policy"]["model_name"] = ckpt["model_name"]
        run_cfg["policy"]["served_model_name"] = ckpt["served_model_name"]
        output_dir = Path("outputs") / cfg["experiment_name"] / f"step_{ckpt['step']}"
        run_cfg["output"]["dir"] = output_dir.as_posix()
        cfg_path = out_root / f"eval_step_{ckpt['step']}.json"
        cfg_path.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        cmd = ["python3", "scripts/eval/run_policy_eval.py", "--config", str(cfg_path)]
        if args.dry_run:
            cmd.append("--dry-run")
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            model_path = Path(ckpt["model_name"])
            if not model_path.is_absolute():
                model_path = ROOT / model_path
            if not (model_path / "config.json").exists():
                raise FileNotFoundError(f"Exported HF checkpoint is missing: {model_path}")

            log_path = out_root / f"assistant_step_{ckpt['step']}.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICES": args.assistant_gpus,
                    "MODEL_PATH": str(model_path),
                    "SERVED_MODEL_NAME": ckpt["served_model_name"],
                    "PORT": "8000",
                    "TP_SIZE": "1",
                    "GPU_MEM_UTIL": args.assistant_gpu_memory_utilization,
                    "MAX_MODEL_LEN": args.assistant_max_model_len,
                    "MAX_NUM_SEQS": "8",
                }
            )
            process = None
            with log_path.open("w", encoding="utf-8") as log_file:
                try:
                    process = subprocess.Popen(
                        ["bash", "scripts/vllm_server/start_assistant_7b.sh"],
                        cwd=ROOT,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    wait_for_endpoint(run_cfg["policy"]["base_url"], process, args.server_timeout)
                    subprocess.run(cmd, cwd=ROOT, check=True)
                finally:
                    stop_process(process)
        manifest.append(
            {
                "step": ckpt["step"],
                "config": cfg_path.relative_to(ROOT).as_posix(),
                "output_dir": run_cfg["output"]["dir"],
            }
        )

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote checkpoint eval manifest to {out_root}")


if __name__ == "__main__":
    main()
