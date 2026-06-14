from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_supervised_example(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
    max_length: int = 8192,
) -> dict[str, Any] | None:
    assistant_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if not assistant_indices:
        return None
    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = _tokenize(tokenizer, full_text)
    except Exception:
        return None
    if len(full_ids) > max_length:
        return None

    labels = [IGNORE_INDEX] * len(full_ids)
    for assistant_idx in assistant_indices:
        try:
            prefix_text = tokenizer.apply_chat_template(
                messages[:assistant_idx],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            with_assistant_text = tokenizer.apply_chat_template(
                messages[: assistant_idx + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = _tokenize(tokenizer, prefix_text)
            with_assistant_ids = _tokenize(tokenizer, with_assistant_text)
        except Exception:
            return None

        start = len(prefix_ids)
        end = min(len(with_assistant_ids), len(full_ids))
        if start >= end:
            continue
        for pos in range(start, end):
            labels[pos] = full_ids[pos]

    if sum(1 for value in labels if value != IGNORE_INDEX) < 5:
        return None
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


class TrajectorySFTDataset:
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        tools: list[dict[str, Any]] | None = None,
        max_length: int = 8192,
    ):
        self.examples: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                example = build_supervised_example(row["messages"], tokenizer, tools=tools, max_length=max_length)
                if example is not None:
                    self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_idx, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row_idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[row_idx, :length] = torch.tensor(item["labels"], dtype=torch.long)
            attention_mask[row_idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
