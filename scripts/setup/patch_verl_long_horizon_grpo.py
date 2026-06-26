#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PATCH_MARKER = "# BEGIN LONG_HORIZON_GRPO_PATCH"
PATCH_END = "# END LONG_HORIZON_GRPO_PATCH"

PATCH_CODE = r'''
# BEGIN LONG_HORIZON_GRPO_PATCH

def _lh_get(config: Any, name: str, default: Any) -> Any:
    try:
        return config.get(name, default)
    except Exception:
        return getattr(config, name, default)


def _lh_global_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted_sum = (values * mask).sum()
    weight = mask.sum().clamp(min=1.0)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            packed = torch.stack([weighted_sum, weight])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            weighted_sum, weight = packed[0], packed[1].clamp(min=1.0)
    except Exception:
        pass
    return weighted_sum / weight


def _lh_balanced_agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    valid = loss_mask.bool()
    if valid.sum() == 0:
        return loss_mat.sum() * 0.0
    seq_mask = valid.any(dim=-1)
    seq_adv_sum = (advantages * loss_mask).sum(dim=-1)
    seq_len = loss_mask.sum(dim=-1).clamp(min=1.0)
    seq_adv = seq_adv_sum / seq_len
    pos_seq = (seq_adv > 0) & seq_mask
    neg_seq = (seq_adv < 0) & seq_mask
    zero_seq = (seq_adv == 0) & seq_mask
    total_seq = (pos_seq.sum() + neg_seq.sum() + zero_seq.sum()).clamp(min=1)
    loss = loss_mat.sum() * 0.0
    if pos_seq.any():
        pos_mask = pos_seq.unsqueeze(-1) & valid
        loss = loss + (pos_seq.sum().float() / total_seq.float()) * loss_mat[pos_mask].mean()
    if neg_seq.any():
        neg_mask = neg_seq.unsqueeze(-1) & valid
        loss = loss + (neg_seq.sum().float() / total_seq.float()) * loss_mat[neg_mask].mean()
    return loss


@register_policy_loss("long_horizon_balanced")
def compute_policy_loss_long_horizon_balanced(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    policy_cfg = _lh_get(config, "policy_loss", {})
    clip_ratio = float(_lh_get(config, "clip_ratio", 0.2))
    clip_ratio_low = _lh_get(config, "clip_ratio_low", clip_ratio)
    clip_ratio_high = _lh_get(config, "clip_ratio_high", clip_ratio)
    clip_ratio_c = float(_lh_get(config, "clip_ratio_c", 3.0))
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    dynamic_enabled = bool(_lh_get(policy_cfg, "dynamic_clip_enabled", True))
    observed_clip_ratio = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
    if dynamic_enabled:
        # These are absolute PPO clip ratios, not half-widths around the ratio center.
        target = float(_lh_get(policy_cfg, "target_clip_ratio", 0.05))
        min_clip_ratio = float(_lh_get(policy_cfg, "min_clip_ratio", _lh_get(policy_cfg, "min_clip_width", 0.05)))
        max_clip_ratio = float(_lh_get(policy_cfg, "max_clip_ratio", _lh_get(policy_cfg, "max_clip_width", 0.4)))
        with torch.no_grad():
            base_clipped = ((ratio < (1.0 - float(clip_ratio_low))) | (ratio > (1.0 + float(clip_ratio_high)))).float()
            observed_clip_ratio = _lh_global_masked_mean(base_clipped, response_mask).detach()
            observed_value = float(observed_clip_ratio.item())
            if observed_value > target:
                width_scale = 1.15
            elif observed_value < target * 0.5:
                width_scale = 0.9
            else:
                width_scale = 1.0
            clip_ratio_low = max(min_clip_ratio, min(max_clip_ratio, float(clip_ratio_low) * width_scale))
            clip_ratio_high = max(min_clip_ratio, min(max_clip_ratio, float(clip_ratio_high) * width_scale))

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - float(clip_ratio_low), 1 + float(clip_ratio_high))
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = _lh_balanced_agg_loss(pg_losses, response_mask, advantages)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/lh_clip_low": float(clip_ratio_low),
        "actor/lh_clip_high": float(clip_ratio_high),
        "actor/lh_balanced_aggregation": 1.0,
        "actor/lh_observed_clip_ratio": observed_clip_ratio.detach().item(),
    }
    return pg_loss, pg_metrics


@register_adv_est("grpo_long_horizon")
def compute_grpo_long_horizon_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    turn_token_weights: torch.Tensor | None = None,
    constraint_violated: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        bsz = scores.shape[0]
        groups: dict[Any, list[int]] = defaultdict(list)
        for i in range(bsz):
            groups[index[i]].append(i)
        scalars = torch.zeros_like(scores)
        min_std = float(_lh_get(config, "lh_min_reward_std", 1e-4)) if config is not None else 1e-4
        smooth_blend = float(_lh_get(config, "lh_smooth_blend", 0.0)) if config is not None else 0.0
        adv_clip = float(_lh_get(config, "lh_adv_clip", 5.0)) if config is not None else 5.0
        singleton_policy = str(_lh_get(config, "lh_singleton_policy", "skip") if config is not None else "skip").lower()
        batch_mean = scores.mean()
        batch_std = scores.std(unbiased=True).clamp(min=epsilon) if scores.numel() > 1 else scores.new_tensor(1.0)
        for _, idxs in groups.items():
            idxs_t = torch.as_tensor(idxs, device=scores.device, dtype=torch.long)
            # Leg 1b — reject: violated rollouts are excluded from the group
            # baseline AND get zero advantage (they neither learn nor pollute the
            # group mean/std). This is the reject half of reject-resample; the
            # resample half (re-generate to refill the batch) lives in the trainer
            # rollout loop and needs the veRL async rollout plumbing.
            if constraint_violated is not None:
                ok = ~constraint_violated[idxs_t].bool()
                valid_t = idxs_t[ok]
            else:
                valid_t = idxs_t
            # Singleton groups have no group-relative baseline. Default is to skip
            # them; set lh_singleton_policy=native to match veRL's original
            # fallback behavior and use the raw score as the scalar advantage.
            if len(valid_t) <= 1:
                if singleton_policy == "native" and len(valid_t) == 1:
                    scalars[valid_t] = scores[valid_t]
                continue
            valid_scores = scores[valid_t]
            group_mean = valid_scores.mean()
            group_std = valid_scores.std(unbiased=True)
            if group_std < min_std:
                continue
            local = (valid_scores - group_mean) / (group_std + epsilon) if norm_adv_by_std_in_grpo else (valid_scores - group_mean)
            if smooth_blend > 0.0:
                # Ablation-only: cross-group/global scaling can break GRPO
                # group-relative semantics when task difficulty varies. Keep units
                # consistent with norm_adv_by_std_in_grpo when this is enabled.
                global_adv = (valid_scores - batch_mean) / batch_std if norm_adv_by_std_in_grpo else (valid_scores - batch_mean)
                local = (1.0 - smooth_blend) * local + smooth_blend * global_adv
            scalars[valid_t] = local
        scalars = torch.clamp(scalars, min=-adv_clip, max=adv_clip)
        if turn_token_weights is not None:
            # Leg 2 — per-turn credit via FIXED token weights (e.g. 2.0 on
            # decision/write-turn tokens, 1.0 elsewhere). Normalize so the mean
            # weight over policy tokens is 1.0: only the *distribution* of credit
            # across turns changes, not the scale. Weights are structural (never
            # learned), so this channel is not hackable.
            mask_f = response_mask.float()
            mean_w = (turn_token_weights.float() * mask_f).sum() / mask_f.sum().clamp(min=1.0)
            norm_w = turn_token_weights.float() / mean_w.clamp(min=1e-8)
            advantages = scalars.unsqueeze(-1) * norm_w * response_mask
        else:
            advantages = scalars.unsqueeze(-1) * response_mask
        returns = advantages
    return advantages, returns

# END LONG_HORIZON_GRPO_PATCH
'''


def patch_core_algos(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        start = text.index(PATCH_MARKER)
        end = text.index(PATCH_END, start) + len(PATCH_END)
        new_text = text[:start].rstrip() + "\n\n" + PATCH_CODE.strip() + "\n" + text[end:]
    else:
        insert_at = text.find("\n\n@deprecated(\"verl.trainer.ppo.core_algos.compute_policy_loss_vanilla\")")
        if insert_at < 0:
            raise RuntimeError("Cannot find insertion point before deprecated compute_policy_loss")
        new_text = text[:insert_at].rstrip() + "\n\n" + PATCH_CODE.strip() + "\n" + text[insert_at:]
    if new_text == text:
        return False
    backup = path.with_suffix(path.suffix + ".long_horizon_backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(new_text, encoding="utf-8")
    return True


def patch_ray_trainer(path: Path) -> bool:
    """Inject turn_token_weights passing into ray_trainer.compute_advantage.

    veRL's generic advantage branch builds ``adv_kwargs`` from a fixed set of
    DataProto fields. We add one optional field so Leg-2 per-turn credit weights
    (produced on the rollout side) reach the advantage estimator. Silently
    no-ops if the veRL version's anchor text has changed.
    """
    text = path.read_text(encoding="utf-8")
    if "LONG_HORIZON_TURN_WEIGHTS" in text:
        return False
    anchor = 'adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]'
    if anchor not in text:
        return False
    inject = (
        anchor + "\n"
        '        if "turn_token_weights" in data.batch:  # LONG_HORIZON_TURN_WEIGHTS\n'
        '            adv_kwargs["turn_token_weights"] = data.batch["turn_token_weights"]\n'
        '        if "constraint_violated" in data.batch:  # LONG_HORIZON_CONSTRAINT_REJECT\n'
        '            adv_kwargs["constraint_violated"] = data.batch["constraint_violated"]'
    )
    backup = path.with_suffix(path.suffix + ".long_horizon_backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(anchor, inject, 1), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch veRL with long-horizon GRPO loss/advantage hooks.")
    parser.add_argument("--verl-path", default=None)
    args = parser.parse_args()
    root = Path(args.verl_path or "../verl").resolve()
    core = root / "verl" / "trainer" / "ppo" / "core_algos.py"
    ray_trainer = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    if not core.exists():
        raise SystemExit(f"Cannot find veRL core_algos.py under {root}")
    changed = patch_core_algos(core)
    print(f"patched={changed} core_algos={core}")
    if ray_trainer.exists():
        changed_rt = patch_ray_trainer(ray_trainer)
        print(f"patched={changed_rt} ray_trainer={ray_trainer} (Leg 2 turn_token_weights)")
    else:
        print(f"skip ray_trainer patch (not found: {ray_trainer})")


if __name__ == "__main__":
    main()
