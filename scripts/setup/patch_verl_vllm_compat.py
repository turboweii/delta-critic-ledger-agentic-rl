#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


OLD = """    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()
"""

NEW = """    async def wait_for_requests_to_drain(self):
        wait_method = getattr(self.engine, "wait_for_requests_to_drain", None)
        if wait_method is not None:
            await wait_method()
            return
        while self.engine.output_processor.has_unfinished_requests():
            await asyncio.sleep(0.01)
"""

ROLLOUT_IMPORT_OLD = """from typing import Any, Optional

import torch
"""

ROLLOUT_IMPORT_NEW = """from typing import Any, Optional

import torch
from omegaconf import open_dict
"""

ROLLOUT_ASSIGN_OLD = """    # Always pass rollout_correction config to actor for metrics computation
    policy_loss_config["rollout_correction"] = rollout_corr_config
"""

ROLLOUT_ASSIGN_NEW = """    # Always pass rollout_correction config to actor for metrics computation.
    # PolicyLossConfig is structured and does not declare this runtime-only field.
    with open_dict(policy_loss_config):
        policy_loss_config["rollout_correction"] = rollout_corr_config
"""

ACTOR_META_OLD = """                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
"""

ACTOR_META_NEW = """                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
"""


def main() -> None:
    import verl

    target = (
        Path(verl.__file__).resolve().parent
        / "workers"
        / "rollout"
        / "vllm_rollout"
        / "vllm_async_server.py"
    )
    text = target.read_text(encoding="utf-8")
    if NEW not in text:
        if OLD not in text:
            raise RuntimeError(
                "Unsupported veRL vLLM server implementation; refusing to patch "
                f"unexpected source: {target}"
            )
        target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        print(f"Applied veRL/vLLM compatibility patch: {target}")
    else:
        print(f"veRL/vLLM compatibility patch already applied: {target}")

    rollout_helper = (
        Path(verl.__file__).resolve().parent
        / "trainer"
        / "ppo"
        / "rollout_corr_helper.py"
    )
    helper_text = rollout_helper.read_text(encoding="utf-8")
    if ROLLOUT_ASSIGN_NEW not in helper_text:
        if ROLLOUT_IMPORT_OLD not in helper_text or ROLLOUT_ASSIGN_OLD not in helper_text:
            raise RuntimeError(
                "Unsupported veRL rollout correction implementation; refusing to patch "
                f"unexpected source: {rollout_helper}"
            )
        helper_text = helper_text.replace(ROLLOUT_IMPORT_OLD, ROLLOUT_IMPORT_NEW, 1)
        helper_text = helper_text.replace(ROLLOUT_ASSIGN_OLD, ROLLOUT_ASSIGN_NEW, 1)
        rollout_helper.write_text(helper_text, encoding="utf-8")
        print(f"Applied veRL rollout-correction config patch: {rollout_helper}")
    else:
        print(f"veRL rollout-correction config patch already applied: {rollout_helper}")

    ray_trainer = (
        Path(verl.__file__).resolve().parent
        / "trainer"
        / "ppo"
        / "ray_trainer.py"
    )
    trainer_text = ray_trainer.read_text(encoding="utf-8")
    if ACTOR_META_NEW not in trainer_text:
        if ACTOR_META_OLD not in trainer_text:
            raise RuntimeError(
                "Unsupported veRL actor update implementation; refusing to patch "
                f"unexpected source: {ray_trainer}"
            )
        ray_trainer.write_text(
            trainer_text.replace(ACTOR_META_OLD, ACTOR_META_NEW, 1),
            encoding="utf-8",
        )
        print(f"Applied veRL async-rollout temperature patch: {ray_trainer}")
    else:
        print(f"veRL async-rollout temperature patch already applied: {ray_trainer}")


if __name__ == "__main__":
    main()
