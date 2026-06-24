from __future__ import annotations

import json
import time
import urllib.request
import copy
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from delta_critic_ledger.prompts import tau_system_prompt


@dataclass
class EvalResult:
    task_id: int
    sample_id: int
    success: bool
    reward: float
    num_turns: int
    num_tool_calls: int
    error: Optional[str]


class OpenAICompatPolicy:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        max_context_chars: int | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars
        self.was_truncated = False
        self.tools: list[dict] = []

    def set_tools(self, tools: list[dict]) -> None:
        self.tools = tools

    def _message_chars(self, message: dict) -> int:
        total = len(str(message.get("content", "") or ""))
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += len(str(fn.get("name", ""))) + len(str(fn.get("arguments", "")))
        total += len(str(message.get("name", "") or ""))
        total += len(str(message.get("tool_call_id", "") or ""))
        return total

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        self.was_truncated = False
        if not self.max_context_chars:
            return messages
        if sum(self._message_chars(m) for m in messages) <= self.max_context_chars:
            return messages

        system = [copy.deepcopy(messages[0])] if messages and messages[0].get("role") == "system" else []
        recent: list[dict] = []
        budget = self.max_context_chars - sum(self._message_chars(m) for m in system)
        if budget <= 0:
            clipped = system or [copy.deepcopy(messages[0])]
            clipped[0]["content"] = str(clipped[0].get("content", ""))[-self.max_context_chars:]
            self.was_truncated = True
            return clipped

        for message in reversed(messages[len(system):]):
            size = self._message_chars(message)
            if size <= budget:
                recent.append(copy.deepcopy(message))
                budget -= size
                continue
            if not recent:
                clipped = copy.deepcopy(message)
                content = str(clipped.get("content", "") or "")
                clipped["content"] = content[-max(0, budget):]
                recent.append(clipped)
            break

        self.was_truncated = True
        return system + list(reversed(recent))

    def __call__(self, messages: list[dict]) -> dict:
        request_messages = self._truncate_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": request_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        return {
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
        }


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return json.loads(str(raw) or "{}")


def run_single_task(env: Any, policy: OpenAICompatPolicy, task_idx: int, max_turns: int) -> EvalResult:
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    obs = env.reset(task_index=task_idx)
    policy.set_tools(env.tools_info)
    messages = [
        {
            "role": "system",
            "content": tau_system_prompt(env.wiki),
        },
        {"role": "user", "content": str(obs.observation)},
    ]
    reward = 0.0
    num_tool_calls = 0
    error = None
    for turn in range(max_turns):
        try:
            assistant = policy(messages)
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc["function"]["name"]
                    args = parse_tool_arguments(tc["function"].get("arguments"))
                    step = env.step(Action(name=fn, kwargs=args))
                    num_tool_calls += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{num_tool_calls}"),
                        "name": fn,
                        "content": str(step.observation),
                    })
                    reward = float(step.reward)
                    if step.done:
                        return EvalResult(task_idx, 0, reward >= 1.0, reward, turn + 1, num_tool_calls, error)
            else:
                step = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant.get("content", "")}))
                reward = float(step.reward)
                if step.done:
                    return EvalResult(task_idx, 0, reward >= 1.0, reward, turn + 1, num_tool_calls, error)
                messages.append({"role": "user", "content": str(step.observation)})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    return EvalResult(task_idx, 0, reward >= 1.0, reward, max_turns, num_tool_calls, error)


def write_eval_report(results: list[EvalResult], output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    success_rate = sum(1 for r in results if r.success) / len(results) if results else 0.0
    error_rate = sum(1 for r in results if r.error) / len(results) if results else 0.0
    by_task: dict[int, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result)
    task_splits: dict[int, str] = {}
    split_file = config.get("env", {}).get("split_file")
    if split_file:
        split_path = Path(split_file)
        if split_path.exists():
            split_data = json.loads(split_path.read_text(encoding="utf-8"))
            task_splits.update({int(task_id): "seen" for task_id in split_data.get("seen_task_ids", [])})
            task_splits.update({int(task_id): "unseen" for task_id in split_data.get("unseen_task_ids", [])})

    per_task = []
    for task_id, items in sorted(by_task.items()):
        successes = sum(1 for item in items if item.success)
        per_task.append({
            "task_id": task_id,
            "split": task_splits.get(task_id, "unspecified"),
            "num_samples": len(items),
            "success_count": successes,
            "pass_at_1": successes / len(items) if items else 0.0,
            f"pass_at_{len(items)}": 1.0 if successes > 0 else 0.0,
            "avg_tool_calls": sum(item.num_tool_calls for item in items) / len(items) if items else 0.0,
            "error_count": sum(1 for item in items if item.error),
        })
    pass_at_1 = success_rate
    max_samples = max((len(items) for items in by_task.values()), default=0)
    pass_at_k = (
        sum(1 for items in by_task.values() if any(item.success for item in items)) / len(by_task)
        if by_task else 0.0
    )
    split_metrics = {}
    for split_name in sorted(set(task_splits.values())):
        split_results = [result for result in results if task_splits.get(result.task_id) == split_name]
        split_tasks = {result.task_id for result in split_results}
        split_metrics[split_name] = {
            "num_tasks": len(split_tasks),
            "num_samples": len(split_results),
            "success_rate": (
                sum(1 for result in split_results if result.success) / len(split_results) if split_results else 0.0
            ),
            f"pass_at_{max_samples}": (
                sum(
                    1
                    for task_id in split_tasks
                    if any(result.success for result in split_results if result.task_id == task_id)
                )
                / len(split_tasks)
                if split_tasks
                else 0.0
            ),
        }
    report = {
        "success_rate": success_rate,
        "pass_at_1": pass_at_1,
        f"pass_at_{max_samples}": pass_at_k,
        "error_rate": error_rate,
        "num_samples": len(results),
        "num_tasks": len(by_task),
        "avg_tool_calls": sum(r.num_tool_calls for r in results) / len(results) if results else 0.0,
        "by_split": split_metrics,
        "per_task": per_task,
        "config": config,
        "results": [asdict(r) for r in results],
    }
    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report
