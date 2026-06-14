#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a veRL LoRA checkpoint into an HF model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True, help="veRL actor directory or its lora_adapter directory.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    adapter = checkpoint if checkpoint.name == "lora_adapter" else checkpoint / "lora_adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"LoRA adapter not found: {adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    merged = model.merge_and_unload()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(output)
    print(f"Exported merged checkpoint to {output}")


if __name__ == "__main__":
    main()

