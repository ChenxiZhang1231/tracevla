# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def compute_decoupled_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    proximal_logprobs: Optional[torch.Tensor] = None,
    versions: Optional[torch.Tensor] = None,
    current_version: Optional[float] = None,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    behave_weight_threshold: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Compute actor loss for decoupled PPO with optional proximal policy anchor."""
    assert logprobs.dtype == torch.float32, (
        "logprobs must be float32 to keep numerical stability"
    )
    assert old_logprobs.dtype == torch.float32, (
        "old_logprobs must be float32 to keep numerical stability"
    )
    assert advantages.dtype == torch.float32, (
        "advantages must be float32 to keep numerical stability"
    )

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if proximal_logprobs is None:
        if versions is None or current_version is None:
            proximal_logprobs = old_logprobs.detach()
        else:
            v_behav = versions.float()
            v_theta = float(current_version)
            v_prox = v_theta - 1.0

            version_diff = v_theta - v_behav
            version_gap = v_prox - v_behav
            generated_tokens_mask = versions >= 0
            alpha = torch.where(
                (version_diff > 0) & generated_tokens_mask,
                version_gap / version_diff,
                torch.zeros_like(v_behav),
            )
            while alpha.dim() < logprobs.dim():
                alpha = alpha.unsqueeze(-1)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            proximal_logprobs = (
                old_logprobs + alpha * (logprobs - old_logprobs)
            ).detach()

    assert proximal_logprobs.dtype == torch.float32, (
        "proximal_logprobs must be float32 to keep numerical stability"
    )

    loss_mask_count = loss_mask.count_nonzero() or 1
    proximal_ratio = torch.where(
        loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0
    )
    clipped_proximal_ratio = torch.clamp(
        proximal_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    pg_loss1 = -advantages * proximal_ratio
    pg_loss2 = -advantages * clipped_proximal_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        pg_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(pg_loss, dtype=torch.bool)

    behav_weight = torch.exp(proximal_logprobs - old_logprobs)
    behav_mask = (
        (behav_weight <= behave_weight_threshold).logical_and(loss_mask)
        if behave_weight_threshold is not None
        else loss_mask
    )
    behav_mask_count = behav_mask.count_nonzero() or 1

    pg_loss = loss_agg_func(pg_loss * behav_weight, behav_mask, loss_mask_ratio)
    if critic_warmup:
        pg_loss = torch.tensor(0.0, device=pg_loss.device)

    with torch.no_grad():
        clip_fraction = (pg_loss1 < pg_loss2).logical_and(
            loss_mask
        ).count_nonzero() / loss_mask_count
        dual_clip_fraction = (
            dual_clip_mask.logical_and(loss_mask).count_nonzero() / loss_mask_count
        )
        proximal_approx_kl = (
            -torch.where(loss_mask, logprobs - proximal_logprobs, 0.0).sum()
            / loss_mask_count
        )
        behav_approx_kl = (
            -torch.where(behav_mask, proximal_logprobs - old_logprobs, 0.0).sum()
            / behav_mask_count
        )
        behav_clip_fraction = 1.0 - (behav_mask_count / loss_mask_count)

    metrics_data = {
        "actor/policy_loss": pg_loss.detach(),
        "actor/proximal_ratio": masked_mean(proximal_ratio.detach(), loss_mask),
        "actor/clipped_proximal_ratio": masked_mean(
            clipped_proximal_ratio.detach(), loss_mask
        ),
        "actor/clip_fraction": clip_fraction,
        "actor/dual_clip_fraction": dual_clip_fraction,
        "actor/behav_clip_fraction": behav_clip_fraction,
        "actor/proximal_approx_kl": proximal_approx_kl,
        "actor/behav_approx_kl": behav_approx_kl,
    }
    if (
        versions is not None
        and current_version is not None
        and versions.shape == loss_mask.shape
        and loss_mask.any()
    ):
        metrics_data["actor/average_version"] = versions[loss_mask].float().mean()
        metrics_data["actor/current_version"] = torch.tensor(
            float(current_version), device=logprobs.device
        )

    return pg_loss, metrics_data


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    clip_log_ratio_min: Optional[float] = None,
    clip_log_ratio_max: Optional[float] = None,
    fast_path_zero_loss_mask: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """
    if fast_path_zero_loss_mask and (
        loss_mask is not None and loss_mask[0].sum() == 0.0
    ):
        return torch.tensor(0.0, device=logprobs.device), {
            "actor/token_num": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss_mbs_mean": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss_abs": torch.tensor(0.0, device=logprobs.device),
            "actor/ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/clipped_ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/dual_cliped_ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/approx_kl": torch.tensor(0.0, device=logprobs.device),
            "actor/clip_fraction": torch.tensor(0.0, device=logprobs.device),
        }

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32, (
        "logprobs must be float32 to keep numerical stability"
    )
    assert old_logprobs.dtype == torch.float32, (
        "old_logprobs must be float32 to keep numerical stability"
    )
    assert advantages.dtype == torch.float32, (
        "advantages must be float32 to keep numerical stability"
    )

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    log_ratio = logprobs - old_logprobs
    if clip_log_ratio_min is not None:
        log_ratio = torch.clamp(log_ratio, min=clip_log_ratio_min)
    if clip_log_ratio_max is not None:
        log_ratio = torch.clamp(log_ratio, max=clip_log_ratio_max)
    ratio = torch.where(loss_mask, torch.exp(log_ratio), 0)
    approx_kl = torch.where(loss_mask, log_ratio.detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio

    clip_mask = policy_loss1.detach() < policy_loss2.detach()

    policy_loss = torch.max(policy_loss1, policy_loss2)
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0"
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    metric_policy_loss_abs = loss_agg_func(
        policy_loss.abs(), loss_mask, loss_mask_ratio
    )
    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask = (dual_clip_mask * loss_mask).bool()

    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask_count)
    approx_kl = -torch.sum(approx_kl) / float(loss_mask_count)

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

    # Compile metrics for logging
    loss_mask_for_metrics = loss_mask
    ratio_for_metrics = ratio.detach()
    ratio_abs_for_metrics = (ratio - 1).abs().detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/policy_loss_abs": metric_policy_loss_abs.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/ratio_abs": masked_mean(ratio_abs_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)

    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach(),
        "critic/value_clip_ratio": value_clip_ratio.detach(),
        "critic/explained_variance": explained_variance.detach(),
    }
    return value_loss, metrics_data


@register_policy_loss("decoupled_actor_critic")
def compute_decoupled_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """Compute decoupled PPO actor+critic loss."""
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_decoupled_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)
    return loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data


def compute_stepwise_ppo_actor_loss(
    stepwise_logprobs: torch.Tensor,
    stepwise_old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Step-wise PPO Actor Loss for Trace-VLA.

    This implements PPO clipping at the denoising step level:
        L_actor = E_t[min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t)]

    where ratio_t = pi_theta(x_{t-1}|x_t) / pi_old(x_{t-1}|x_t)

    Args:
        stepwise_logprobs (torch.Tensor): Current logprobs. Shape: [B, T, action_chunk, action_dim]
        stepwise_old_logprobs (torch.Tensor): Old logprobs. Shape: [B, T, action_chunk, action_dim]
        advantages (torch.Tensor): Step-level advantages. Shape: [B, T]
        clip_ratio_low (float): Lower clipping bound.
        clip_ratio_high (float): Upper clipping bound.
        loss_mask (torch.Tensor, optional): Mask for valid samples. Shape: [B] or [B, T]

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """
    # Sum logprobs over action dimensions to get joint probability
    # [B, T, action_chunk, action_dim] -> [B, T]
    if stepwise_logprobs.dim() == 4:
        logprobs_sum = stepwise_logprobs.sum(dim=(-1, -2))
        old_logprobs_sum = stepwise_old_logprobs.sum(dim=(-1, -2))
    elif stepwise_logprobs.dim() == 3:
        logprobs_sum = stepwise_logprobs.sum(dim=-1)
        old_logprobs_sum = stepwise_old_logprobs.sum(dim=-1)
    else:
        logprobs_sum = stepwise_logprobs
        old_logprobs_sum = stepwise_old_logprobs

    # Compute log ratio with numerical stability
    log_ratio = logprobs_sum - old_logprobs_sum
    log_ratio = log_ratio.clamp(min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio)

    # PPO Clipping
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    # Policy loss (negative because we maximize)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio
    policy_loss = torch.max(policy_loss1, policy_loss2)

    # Handle loss mask
    if loss_mask is not None:
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(-1).expand_as(policy_loss)
        policy_loss = (policy_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    else:
        policy_loss = policy_loss.mean()

    # Compute metrics
    with torch.no_grad():
        approx_kl = log_ratio.mean()
        clip_fraction = ((policy_loss1 < policy_loss2).float()).mean()
        ratio_mean = ratio.mean()

    metrics = {
        "actor/stepwise_policy_loss": policy_loss.detach(),
        "actor/stepwise_approx_kl": approx_kl.detach(),
        "actor/stepwise_clip_fraction": clip_fraction.detach(),
        "actor/stepwise_ratio": ratio_mean.detach(),
    }

    return policy_loss, metrics


def compute_stepwise_critic_loss(
    stepwise_values: torch.Tensor,
    returns: torch.Tensor,
    stepwise_prev_values: torch.Tensor,
    value_clip: float = 0.2,
    huber_delta: float = 10.0,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Step-wise Critic Loss for Trace-VLA.

    This implements value clipping at the denoising step level:
        L_value = E_t[(V_phi(x_hat_{0|t}, t) - R_chunk)^2]

    Note: All denoising steps share the same target (chunk-level reward).

    Args:
        stepwise_values (torch.Tensor): Current value predictions. Shape: [B, T]
        returns (torch.Tensor): Target values (chunk reward broadcast). Shape: [B, T]
        stepwise_prev_values (torch.Tensor): Old value predictions. Shape: [B, T]
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.
        loss_mask (torch.Tensor, optional): Mask for valid samples. Shape: [B] or [B, T]

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    # Value clipping (PPO style)
    value_pred_clipped = stepwise_prev_values + (
        stepwise_values - stepwise_prev_values
    ).clamp(-value_clip, value_clip)

    # Huber Loss
    value_loss_original = huber_loss(returns - stepwise_values, huber_delta)
    value_loss_clipped = huber_loss(returns - value_pred_clipped, huber_delta)
    value_loss = torch.max(value_loss_original, value_loss_clipped)

    # Average over all denoising steps
    if loss_mask is not None:
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(-1).expand_as(value_loss)
        value_loss = (value_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    else:
        value_loss = value_loss.mean()

    # Compute metrics
    with torch.no_grad():
        returns_var = returns.var().clamp(min=1e-8)
        residual_var = (returns - stepwise_values).var()
        explained_var = 1 - residual_var / returns_var

    metrics = {
        "critic/stepwise_value_loss": value_loss.detach(),
        "critic/stepwise_explained_variance": explained_var.detach(),
    }

    return value_loss, metrics


@register_policy_loss("stepwise_actor_critic")
def compute_stepwise_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute Step-wise PPO Actor+Critic Loss for Trace-VLA.

    This is the core loss function for Trace-VLA that computes both actor
    and critic losses at the denoising step level.

    Expected kwargs:
        - stepwise_logprobs: [B, T, action_chunk, action_dim]
        - stepwise_old_logprobs: [B, T, action_chunk, action_dim]
        - stepwise_values: [B, T]
        - stepwise_prev_values: [B, T]
        - advantages: [B, T]
        - returns: [B, T]
        - clip_ratio_low, clip_ratio_high, value_clip, huber_delta
        - loss_mask: [B] or [B, T]

    Returns:
        Tuple[torch.Tensor, Dict]: (total_loss, metrics_dict)
    """
    actor_loss, actor_metrics = compute_stepwise_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics = compute_stepwise_critic_loss(**kwargs)

    total_loss = actor_loss + critic_loss

    metrics = {**actor_metrics, **critic_metrics}
    metrics["loss/stepwise_total"] = total_loss.detach()

    return total_loss, metrics


@register_policy_loss("hierarchical_actor_critic")
def compute_hierarchical_actor_critic_loss(
    # Chunk-level inputs
    chunk_values: torch.Tensor = None,
    chunk_prev_values: torch.Tensor = None,
    chunk_advantages: torch.Tensor = None,
    chunk_returns: torch.Tensor = None,
    # Step-level inputs
    stepwise_logprobs: torch.Tensor = None,
    stepwise_old_logprobs: torch.Tensor = None,
    stepwise_values: torch.Tensor = None,
    stepwise_prev_values: torch.Tensor = None,
    step_advantages: torch.Tensor = None,
    step_returns: torch.Tensor = None,
    # Loss hyperparameters
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    clip_ratio_c: float = None,
    value_clip: float = 0.2,
    huber_delta: float = 10.0,
    # Coefficients
    chunk_value_coef: float = 0.5,
    step_value_coef: float = 1.0,
    chunk_actor_coef: float = 0.0,
    step_actor_coef: float = 1.0,
    use_bilevel_actor: bool = False,
    # Step weighting
    step_weights: torch.Tensor = None,
    uncertainty_mode: str = "linear",
    use_uncertainty_weighting: bool = False,
    # Masking and scaling
    loss_mask: torch.Tensor = None,
    loss_mask_sum: torch.Tensor = None,
    max_episode_steps: int = None,
    critic_warmup: bool = False,
    chunk_value_warmup: bool = False,  # Phase 1: only chunk_value trains (skip step_value)
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Hierarchical Actor-Critic Loss for FlowRL/HUA-RL.

    This loss function supports:
    1. Chunk-level critic loss (value clipping with huber loss)
    2. Chunk-level actor loss (PPO clipping with per-dim ratio)
    3. Step-level actor loss (PPO with optional uncertainty weighting)
    4. Step-level critic loss (with optional uncertainty weighting)
    """
    metrics = {}
    device = chunk_values.device if chunk_values is not None else stepwise_values.device
    total_loss = torch.tensor(0.0, device=device)

    # Get dimensions
    if stepwise_logprobs is not None:
        if stepwise_logprobs.dim() == 4:
            B, num_chunks, T = stepwise_logprobs.shape[:3]
        else:
            B = stepwise_logprobs.shape[0]
            num_chunks = 1
            T = stepwise_logprobs.shape[1] if stepwise_logprobs.dim() > 1 else 1

    # Determine loss aggregation function
    use_loss_scaling = (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    )
    if use_loss_scaling:
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
    else:
        loss_mask_ratio = None

    # Prepare chunk_loss_mask (reduce to [B] if needed)
    chunk_loss_mask = loss_mask
    chunk_loss_mask_ratio = loss_mask_ratio
    if loss_mask is not None and loss_mask.dim() > 1:
        chunk_loss_mask = loss_mask.any(dim=-1).float()
        if loss_mask_ratio is not None:
            chunk_loss_mask_ratio = loss_mask_ratio.float().mean(dim=-1)

    # ===== 1. Chunk-Level Critic Loss (reuse baseline) =====
    # Note: Chunk critic loss is computed even during warmup (to stabilize chunk_value first)
    if chunk_values is not None and chunk_returns is not None:
        # Prepare chunk_loss_mask_sum for proper scaling
        chunk_loss_mask_sum = None
        if loss_mask_sum is not None:
            # Convert to float for mean computation
            loss_mask_sum_float = loss_mask_sum.float()
            if loss_mask_sum_float.dim() > 1:
                chunk_loss_mask_sum = loss_mask_sum_float.mean(dim=-1)
            else:
                chunk_loss_mask_sum = loss_mask_sum_float

        chunk_critic_loss, chunk_critic_metrics = compute_ppo_critic_loss(
            values=chunk_values,
            returns=chunk_returns,
            prev_values=chunk_prev_values,
            value_clip=value_clip,
            huber_delta=huber_delta,
            loss_mask=chunk_loss_mask.bool() if chunk_loss_mask is not None else None,
            max_episode_steps=max_episode_steps,
            loss_mask_sum=chunk_loss_mask_sum,
        )

        total_loss = total_loss + chunk_value_coef * chunk_critic_loss
        # Rename metrics: critic/* -> critic/chunk_*
        for k, v in chunk_critic_metrics.items():
            metrics[k.replace("critic/", "critic/chunk_")] = v

    # ===== 2. Chunk-Level Actor Loss (reuse baseline) =====
    if use_bilevel_actor and chunk_advantages is not None and chunk_actor_coef > 0 and not critic_warmup:
        # Get chunk logprobs (may be passed directly or derived from stepwise)
        chunk_logprobs = kwargs.get("chunk_logprobs", None)
        chunk_old_logprobs = kwargs.get("chunk_old_logprobs", None)

        if chunk_logprobs is not None and chunk_old_logprobs is not None:
            chunk_logprobs_input = chunk_logprobs
            chunk_old_logprobs_input = chunk_old_logprobs
        elif stepwise_logprobs is not None:
            # Derive from stepwise logprobs
            if stepwise_logprobs.dim() == 4:
                chunk_logprobs_input = stepwise_logprobs.mean(dim=1)
                chunk_old_logprobs_input = stepwise_old_logprobs.mean(dim=1)
            else:
                chunk_logprobs_input = stepwise_logprobs
                chunk_old_logprobs_input = stepwise_old_logprobs
        else:
            raise ValueError("No logprobs available for chunk actor loss")

        # Sum logprobs over action dimensions (like baseline's chunk_level)
        chunk_logprobs_sum = chunk_logprobs_input.sum(dim=list(range(1, chunk_logprobs_input.dim())))
        chunk_old_logprobs_sum = chunk_old_logprobs_input.sum(dim=list(range(1, chunk_old_logprobs_input.dim())))

        # Get advantages
        if chunk_advantages.dim() == 1:
            chunk_adv = chunk_advantages
        else:
            chunk_adv = chunk_advantages.mean(dim=-1) if chunk_advantages.dim() > 1 else chunk_advantages

        # Prepare chunk_loss_mask_sum for proper scaling
        chunk_loss_mask_sum = None
        if loss_mask_sum is not None:
            # Convert to float for mean computation
            loss_mask_sum_float = loss_mask_sum.float()
            if loss_mask_sum_float.dim() > 1:
                chunk_loss_mask_sum = loss_mask_sum_float.mean(dim=-1)
            else:
                chunk_loss_mask_sum = loss_mask_sum_float

        # Reuse baseline actor loss
        chunk_actor_loss, chunk_actor_metrics = compute_ppo_actor_loss(
            logprobs=chunk_logprobs_sum.float(),
            old_logprobs=chunk_old_logprobs_sum.float(),
            advantages=chunk_adv.float(),
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            loss_mask=chunk_loss_mask.bool() if chunk_loss_mask is not None else None,
            max_episode_steps=max_episode_steps,
            loss_mask_sum=chunk_loss_mask_sum,
            critic_warmup=False,  # Already checked above
        )

        total_loss = total_loss + chunk_actor_coef * chunk_actor_loss
        # Rename metrics: actor/* -> actor/chunk_*
        for k, v in chunk_actor_metrics.items():
            metrics[k.replace("actor/", "actor/chunk_")] = v

    # ===== 3. Step-Level Critic Loss (optional) =====
    # Note: Skip during chunk_value_warmup (phase 1: only chunk_value trains)
    # Phase 2 (critic_warmup but not chunk_value_warmup): step_value trains with chunk_value
    if step_value_coef > 0 and stepwise_values is not None and step_returns is not None and not chunk_value_warmup:
        # Flatten for processing: [B, num_chunks, T] -> [B*num_chunks, T] or similar
        stepwise_values_flat = stepwise_values.reshape(-1, stepwise_values.shape[-1]) if stepwise_values.dim() > 2 else stepwise_values
        stepwise_prev_values_flat = stepwise_prev_values.reshape(-1, stepwise_prev_values.shape[-1]) if stepwise_prev_values.dim() > 2 else stepwise_prev_values
        step_returns_flat = step_returns.reshape(-1, step_returns.shape[-1]) if step_returns.dim() > 2 else step_returns

        step_pred_clipped = stepwise_prev_values_flat + (
            stepwise_values_flat - stepwise_prev_values_flat
        ).clamp(-value_clip, value_clip)

        step_loss_orig = huber_loss(step_returns_flat - stepwise_values_flat, huber_delta)
        step_loss_clip = huber_loss(step_returns_flat - step_pred_clipped, huber_delta)
        step_critic_loss_per_step = torch.max(step_loss_orig, step_loss_clip)

        # Apply uncertainty weighting if enabled
        if use_uncertainty_weighting and step_weights is not None:
            step_weights_flat = step_weights.reshape(-1, step_weights.shape[-1]) if step_weights.dim() > 2 else step_weights
            if step_weights_flat.shape != step_critic_loss_per_step.shape:
                step_weights_flat = step_weights_flat.expand_as(step_critic_loss_per_step)
            weighted_step_critic = step_critic_loss_per_step * step_weights_flat
            weight_sum = step_weights_flat.sum().clamp(min=1)
            step_critic_loss = weighted_step_critic.sum() / weight_sum
            metrics["uncertainty/hierarchical_weight_mean"] = step_weights_flat.mean().detach()
            metrics["uncertainty/hierarchical_weight_std"] = step_weights_flat.std().detach()
        else:
            step_critic_loss = step_critic_loss_per_step.mean()

        total_loss = total_loss + step_value_coef * step_critic_loss
        metrics["critic/hierarchical_step_value_loss"] = step_critic_loss.detach()

        # Compute explained variance
        with torch.no_grad():
            returns_var = step_returns_flat.var()
            if returns_var > 1e-8:
                residual_var = (step_returns_flat - stepwise_values_flat).var()
                explained_var = 1 - residual_var / returns_var
                metrics["critic/hierarchical_explained_variance"] = explained_var.detach()

    # ===== 4. Step-Level Actor Loss (hierarchical) =====
    # Note: Skip during critic_warmup to let chunk_value stabilize first
    if step_actor_coef > 0 and stepwise_logprobs is not None and step_advantages is not None and not critic_warmup:
        # Flatten logprobs: [B, num_chunks, T, action_chunk, action_dim] -> [B*num_chunks, T, ...]
        stepwise_logprobs_flat = stepwise_logprobs.reshape(-1, *stepwise_logprobs.shape[-3:]) if stepwise_logprobs.dim() > 4 else stepwise_logprobs
        stepwise_old_logprobs_flat = stepwise_old_logprobs.reshape(-1, *stepwise_old_logprobs.shape[-3:]) if stepwise_old_logprobs.dim() > 4 else stepwise_old_logprobs
        step_advantages_flat = step_advantages.reshape(-1, step_advantages.shape[-1]) if step_advantages.dim() > 2 else step_advantages

        # Sum logprobs over action dimensions: [..., T, action_chunk, action_dim] -> [..., T]
        logprobs_sum = stepwise_logprobs_flat.sum(dim=(-1, -2)) if stepwise_logprobs_flat.dim() >= 3 else stepwise_logprobs_flat.sum(dim=-1)
        old_logprobs_sum = stepwise_old_logprobs_flat.sum(dim=(-1, -2)) if stepwise_old_logprobs_flat.dim() >= 3 else stepwise_old_logprobs_flat.sum(dim=-1)

        log_ratio = (logprobs_sum - old_logprobs_sum).clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

        policy_loss1 = -step_advantages_flat * ratio
        policy_loss2 = -step_advantages_flat * clipped_ratio
        policy_loss_per_step = torch.max(policy_loss1, policy_loss2)

        # Apply uncertainty weighting if enabled
        if use_uncertainty_weighting and step_weights is not None:
            step_weights_flat = step_weights.reshape(-1, step_weights.shape[-1]) if step_weights.dim() > 2 else step_weights
            if step_weights_flat.shape != policy_loss_per_step.shape:
                step_weights_flat = step_weights_flat.expand_as(policy_loss_per_step)
            weighted_policy_loss = policy_loss_per_step * step_weights_flat
            weight_sum = step_weights_flat.sum().clamp(min=1)
            actor_loss = weighted_policy_loss.sum() / weight_sum
        else:
            if loss_mask is not None:
                flat_mask = loss_mask.reshape(-1) if loss_mask.dim() > 1 else loss_mask
                if flat_mask.shape[0] == policy_loss_per_step.shape[0]:
                    flat_mask = flat_mask.unsqueeze(-1).expand_as(policy_loss_per_step)
                actor_loss = (policy_loss_per_step * flat_mask).sum() / flat_mask.sum().clamp(min=1)
            else:
                actor_loss = policy_loss_per_step.mean()

        total_loss = total_loss + step_actor_coef * actor_loss
        metrics["actor/hierarchical_policy_loss"] = actor_loss.detach()

        with torch.no_grad():
            metrics["actor/hierarchical_approx_kl"] = log_ratio.mean().detach()
            metrics["actor/hierarchical_clip_fraction"] = (policy_loss1 < policy_loss2).float().mean().detach()
            metrics["actor/hierarchical_ratio"] = ratio.mean().detach()

    metrics["loss/hierarchical_total"] = total_loss.detach()
    return total_loss, metrics

