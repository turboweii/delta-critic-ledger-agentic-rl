from __future__ import annotations

import argparse
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required for interface audit: {exc}")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise AssertionError(f"{path} must contain a YAML mapping.")
    return data


def load_yaml_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_grpo_config(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    merged: dict[str, Any] = {}
    for item in cfg.get("defaults", []):
        if isinstance(item, str) and item != "_self_" and not item.startswith("ppo_trainer"):
            default_path = path.parent / f"{item}.yaml"
            if default_path.exists():
                merged = deep_merge(merged, load_grpo_config(default_path))
    return deep_merge(merged, cfg)


def assert_tool_interfaces(tool_config: Path) -> None:
    import delta_critic_ledger.verl_integration.tools as tool_mod

    cfg = load_yaml(tool_config)
    tools = cfg.get("tools")
    if not isinstance(tools, list) or not tools:
        raise AssertionError(f"{tool_config} must define a non-empty tools list.")

    for item in tools:
        class_name = item.get("class_name")
        schema = item.get("tool_schema", {})
        function = schema.get("function", {})
        tool_name = function.get("name")
        if not class_name or not tool_name:
            raise AssertionError(f"Malformed tool entry: {item}")
        short_class_name = str(class_name).rsplit(".", 1)[-1]
        if not hasattr(tool_mod, short_class_name):
            raise AssertionError(f"Missing tool class {class_name} for {tool_name}.")

        cls = getattr(tool_mod, short_class_name)
        for method in ("create", "execute", "calc_reward", "release", "get_openai_tool_schema"):
            if not hasattr(cls, method):
                raise AssertionError(f"{class_name} is missing {method}.")
        execute_sig = inspect.signature(cls.execute)
        required = ["self", "instance_id", "parameters"]
        actual = list(execute_sig.parameters)
        if actual[:3] != required:
            raise AssertionError(f"{class_name}.execute signature starts with {actual[:3]}, expected {required}.")
        if not inspect.iscoroutinefunction(cls.execute):
            raise AssertionError(f"{class_name}.execute must be async.")


def assert_interaction_interface(interaction_config: Path) -> None:
    from delta_critic_ledger.verl_integration.interaction import DeltaTauBenchInteraction

    cfg = load_yaml(interaction_config)
    entries = cfg.get("interaction", [])
    if not isinstance(entries, list) or not entries:
        raise AssertionError(f"{interaction_config} must contain an interaction list.")
    entry = entries[0]
    class_name = entry.get("class_name", "")
    if not class_name.endswith("DeltaTauBenchInteraction"):
        raise AssertionError(f"{interaction_config} class_name should point to DeltaTauBenchInteraction.")

    for method in (
        "start_interaction",
        "get_initial_response",
        "generate_response",
        "calculate_score",
        "finalize_interaction",
    ):
        attr = getattr(DeltaTauBenchInteraction, method, None)
        if attr is None:
            raise AssertionError(f"DeltaTauBenchInteraction is missing {method}.")
        if not inspect.iscoroutinefunction(attr):
            raise AssertionError(f"DeltaTauBenchInteraction.{method} must be async.")
    if not hasattr(DeltaTauBenchInteraction, "is_done"):
        raise AssertionError("DeltaTauBenchInteraction is missing is_done.")

    gen_sig = inspect.signature(DeltaTauBenchInteraction.generate_response)
    if list(gen_sig.parameters)[:3] != ["self", "instance_id", "messages"]:
        raise AssertionError("generate_response must accept self, instance_id, messages first.")

    inner_cfg = entry.get("config", {})
    user_model = str(inner_cfg.get("user_model", ""))
    if "32B" not in user_model or "AWQ" not in user_model:
        raise AssertionError("8x4090 interaction user_model should point to the 32B AWQ simulator.")

    delta_cfg = inner_cfg.get("delta_critic", {})
    for key in ("beta_delta", "beta_evidence", "max_trace_steps"):
        if key not in delta_cfg:
            raise AssertionError(f"delta_critic.{key} missing in {interaction_config}.")


def assert_grpo_config(grpo_config: Path, expected_gpus: int) -> None:
    cfg = load_grpo_config(grpo_config)

    actor_rollout_ref = cfg.get("actor_rollout_ref", {})
    rollout = actor_rollout_ref.get("rollout", {})
    actor = actor_rollout_ref.get("actor", {})
    data = cfg.get("data", {})
    trainer = cfg.get("trainer", {})
    algorithm = cfg.get("algorithm", {})
    multi_turn = rollout.get("multi_turn", {})
    agent = rollout.get("agent", {})
    reward_function = cfg.get("custom_reward_function", {})

    if rollout.get("calculate_log_probs") is not True:
        raise AssertionError("rollout.calculate_log_probs must be true for bypass_mode log-prob reuse.")
    if algorithm.get("rollout_correction", {}).get("bypass_mode") is not True:
        raise AssertionError("algorithm.rollout_correction.bypass_mode must be true.")
    if "interaction_config_path" not in multi_turn:
        raise AssertionError("rollout.multi_turn.interaction_config_path missing.")
    if "tool_config_path" not in multi_turn:
        raise AssertionError("rollout.multi_turn.tool_config_path missing.")
    if agent.get("default_agent_loop") != "tau_bench_tool_agent":
        raise AssertionError("rollout.agent.default_agent_loop must inject tau-bench's initial user message.")
    agent_config = ROOT / str(agent.get("agent_loop_config_path", ""))
    if not agent_config.is_file():
        raise AssertionError(f"Custom agent loop config is missing: {agent_config}")
    agent_cfg = load_yaml_any(agent_config)
    agent_entries = agent_cfg if isinstance(agent_cfg, list) else [agent_cfg]
    agent_entry = next((item for item in agent_entries if item.get("name") == "tau_bench_tool_agent"), {})
    adaptive_entropy = agent_entry.get("adaptive_entropy", {})
    if adaptive_entropy.get("enabled") is not True:
        raise AssertionError("tau_bench_tool_agent.adaptive_entropy.enabled should be true.")
    if float(adaptive_entropy.get("min_temperature", 0.0)) >= float(adaptive_entropy.get("max_temperature", 0.0)):
        raise AssertionError("adaptive entropy temperature bounds are invalid.")
    if not str(reward_function.get("path", "")).endswith("verl_integration/reward.py"):
        raise AssertionError("custom_reward_function.path must point to the Delta/Ledger reward adapter.")
    if reward_function.get("name") != "compute_score":
        raise AssertionError("custom_reward_function.name must be compute_score.")
    from delta_critic_ledger.verl_integration.reward import compute_score

    reward = compute_score(
        data_source="tau_bench_airline",
        solution_str="",
        ground_truth="",
        extra_info={"final_score": 1.25, "score_info": {"outcome_score": 1.0}},
    )
    if reward.get("score") != 1.25 or reward.get("outcome_score") != 1.0:
        raise AssertionError("Custom reward adapter did not preserve the interaction score.")
    if int(trainer.get("n_gpus_per_node", 0)) != expected_gpus:
        raise AssertionError(f"trainer.n_gpus_per_node must be {expected_gpus}.")
    if not actor.get("use_kl_loss", False):
        raise AssertionError("actor.use_kl_loss should be enabled for stable GRPO.")
    train_batch_size = int(data.get("train_batch_size", 0))
    mini_batch_size = int(actor.get("ppo_mini_batch_size", 0))
    rollout_n = int(rollout.get("n", 1))
    if mini_batch_size > train_batch_size:
        raise AssertionError(
            f"actor.ppo_mini_batch_size={mini_batch_size} exceeds data.train_batch_size={train_batch_size}."
        )
    if train_batch_size * rollout_n % expected_gpus != 0:
        raise AssertionError("Expanded rollout batch must be divisible by trainer.n_gpus_per_node.")
    if mini_batch_size * rollout_n % expected_gpus != 0:
        raise AssertionError("Expanded PPO mini-batch must be divisible by trainer.n_gpus_per_node.")
    required_context = int(data.get("max_prompt_length", 0)) + int(data.get("max_response_length", 0))
    if int(rollout.get("max_model_len", required_context)) < required_context:
        raise AssertionError("rollout.max_model_len must cover max_prompt_length + max_response_length.")

    adaptive_cfg = load_yaml(ROOT / "configs/train/grpo/adaptive_kl_entropy.yaml")
    if adaptive_cfg.get("enabled") is not True:
        raise AssertionError("adaptive_kl_entropy.yaml should be enabled by default.")
    if float(adaptive_cfg.get("actor_kl_loss_min", 0.0)) > float(adaptive_cfg.get("actor_kl_loss_base", 0.0)):
        raise AssertionError("adaptive KL minimum exceeds base.")
    if float(adaptive_cfg.get("actor_kl_loss_max", 0.0)) < float(adaptive_cfg.get("actor_kl_loss_base", 0.0)):
        raise AssertionError("adaptive KL maximum is below base.")


def assert_grpo_data_contract(interaction_config: Path) -> None:
    spec = importlib.util.spec_from_file_location("build_grpo_parquet", ROOT / "scripts/train/grpo/build_grpo_parquet.py")
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load build_grpo_parquet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = load_yaml(interaction_config)
    registered = {entry["name"] for entry in cfg.get("interaction", [])}
    row = module.build_row(task_id=0, index=0, split="seen")
    interaction_name = row["extra_info"]["interaction_kwargs"]["name"]
    if interaction_name not in registered:
        raise AssertionError(
            f"GRPO parquet interaction name {interaction_name!r} is not registered in {interaction_config}."
        )
    if row["data_source"] != "tau_bench_airline":
        raise AssertionError("GRPO data_source must stay tau_bench_airline for the custom reward adapter.")


def assert_model_config(model_config: Path, expected_user_fragment: str) -> None:
    cfg = load_yaml(model_config)
    user = cfg.get("user", {})
    teacher = cfg.get("teacher", {})
    model_name = str(user.get("model_name", ""))
    if expected_user_fragment not in model_name:
        raise AssertionError(f"user.model_name={model_name!r} does not contain {expected_user_fragment!r}.")
    if str(user.get("quantization", "")).lower() != "awq":
        raise AssertionError("8x4090 user simulator should use AWQ quantization.")
    teacher_name = str(teacher.get("model_name", ""))
    if expected_user_fragment not in teacher_name:
        raise AssertionError(f"teacher.model_name={teacher_name!r} does not contain {expected_user_fragment!r}.")
    if str(teacher.get("quantization", "")).lower() != "awq":
        raise AssertionError("8x4090 teacher policy should use AWQ quantization.")
    if int(user.get("port", 0)) == int(teacher.get("port", 0)):
        raise AssertionError("teacher and user simulator ports must be different.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local tau-bench/veRL interface assumptions.")
    parser.add_argument("--expected-gpus", type=int, default=6)
    parser.add_argument(
        "--grpo-config",
        default="configs/train/grpo/delta_ledger_grpo_8x4090_32b_user.yaml",
        help="GRPO config path relative to the project root, or an absolute path.",
    )
    args = parser.parse_args()

    grpo_config = Path(args.grpo_config)
    if not grpo_config.is_absolute():
        grpo_config = ROOT / grpo_config

    assert_tool_interfaces(ROOT / "configs/tool_config/tau_bench_airline_tools.yaml")
    assert_interaction_interface(ROOT / "configs/interaction_config/tau_bench_airline_delta_ledger.yaml")
    assert_grpo_data_contract(ROOT / "configs/interaction_config/tau_bench_airline_delta_ledger.yaml")
    assert_grpo_config(grpo_config, args.expected_gpus)
    assert_model_config(ROOT / "configs/models/8x4090_qwen.yaml", "32B")
    print("interface_audit: ok")


if __name__ == "__main__":
    main()
