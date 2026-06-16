#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir, load_config
from delta_critic_ledger.sft_dataset import AssistantOnlyCollator, TrajectorySFTDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train" / "sft" / "sft_airline_lora_4x4090.yaml"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_7B", ""))
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write manifest without training.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["name_or_path"] = args.model_path

    output_dir = ensure_dir(ROOT / cfg["output"]["dir"])
    manifest_path = ROOT / cfg["output"]["manifest"]
    ensure_dir(manifest_path.parent)

    if args.dry_run:
        manifest = {
            "status": "dry_run",
            "note": "Config validated. Run without --dry-run on the GPU server to train LoRA weights.",
            "config": cfg,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote dry-run manifest: {manifest_path}")
        return

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "SFT training requires torch, transformers, datasets, and peft. "
            "Install them on the GPU server or rerun with --dry-run."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tool_schemas = None
    tool_config_path = cfg["data"].get("tool_config_path")
    if tool_config_path:
        import yaml

        tool_config = yaml.safe_load((ROOT / tool_config_path).read_text(encoding="utf-8"))
        tool_schemas = [item["tool_schema"] for item in tool_config["tools"]]

    train_dataset = TrajectorySFTDataset(
        ROOT / cfg["data"]["train_jsonl"],
        tokenizer=tokenizer,
        tools=tool_schemas,
        max_length=int(cfg["data"]["max_length"]),
    )
    if len(train_dataset) == 0:
        raise RuntimeError("SFT train dataset is empty after assistant-only masking/length filtering.")

    eval_jsonl = ROOT / cfg["data"].get("eval_jsonl", "")
    eval_dataset = None
    if eval_jsonl.exists() and eval_jsonl.stat().st_size > 0:
        eval_dataset = TrajectorySFTDataset(
            eval_jsonl,
            tokenizer=tokenizer,
            tools=tool_schemas,
            max_length=int(cfg["data"]["max_length"]),
        )
        if len(eval_dataset) == 0:
            eval_dataset = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=cfg["model"].get("attn_impl", "flash_attention_2"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(cfg["train"]["num_epochs"]),
        "per_device_train_batch_size": int(cfg["train"]["per_device_batch_size"]),
        "gradient_accumulation_steps": int(cfg["train"]["grad_accum_steps"]),
        "learning_rate": float(cfg["train"]["lr"]),
        "warmup_ratio": float(cfg["train"]["warmup_ratio"]),
        "logging_steps": int(cfg["train"]["logging_steps"]),
        "save_strategy": cfg["train"]["save_strategy"],
        "bf16": True,
        "report_to": [],
        "ddp_find_unused_parameters": False,
    }
    eval_key = "eval_strategy" if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames else "evaluation_strategy"
    training_kwargs[eval_key] = "epoch" if eval_dataset is not None else "no"
    args_train = TrainingArguments(**training_kwargs)
    data_collator = AssistantOnlyCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output_dir))
        merged_dir = ensure_dir(ROOT / cfg["output"]["merged_dir"])
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "trained",
                    "output_dir": str(output_dir),
                    "merged_dir": str(merged_dir),
                    "config": cfg,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
        print(f"SFT LoRA saved to {output_dir}; merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
