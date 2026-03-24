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

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len+1, bsz] (includes bootstrap).
        dones (torch.Tensor): Done flags (1 if episode ended, else 0). Shape: [seq_len+1, bsz].
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    # Debug: log input shapes (commented out)
    # import logging
    # logger = logging.getLogger(__name__)
    # logger.info(f"[DEBUG] compute_gae_advantages_and_returns: rewards={rewards.shape}, "
    #             f"values={values.shape if values is not None else None}, "
    #             f"dones={dones.shape if dones is not None else None}")

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    # === BASELINE GAE BREAKPOINT 1: 输入 ===
    # 检查: rewards.shape, values.shape, dones.shape, gamma, gae_lambda
    # print(f"\n[BASELINE GAE] Input shapes: rewards={rewards.shape}, values={values.shape if values is not None else None}, dones={dones.shape if dones is not None else None}")
    # print(f"[BASELINE GAE] gamma={gamma}, gae_lambda={gae_lambda}")
    # breakpoint()

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

        # === BASELINE GAE BREAKPOINT 2: 第一次迭代 (step=T-1) 检查 delta 和 gae ===
        # if step == rewards.shape[0] - 1:
        #     print(f"[BASELINE GAE] First iter (step={step}): delta.shape={delta.shape}, delta[:3]={delta.flatten()[:3].tolist()}")
        #     print(f"[BASELINE GAE] gae type={type(gae)}, returns[{step}][:3]={returns[step].flatten()[:3].tolist()}")
        #     breakpoint()

    advantages = returns - values[:-1] if not critic_free else returns

    # === BASELINE GAE BREAKPOINT 3: 输出 (归一化前) ===
    # print(f"\n[BASELINE GAE] Output (pre-norm): advantages.shape={advantages.shape}, returns.shape={returns.shape}")
    # print(f"[BASELINE GAE] advantages[:3]={advantages.flatten()[:3].tolist()}, returns[:3]={returns.flatten()[:3].tolist()}")
    # print(f"[BASELINE GAE] advantages.mean()={advantages.mean().item():.4f}, advantages.std()={advantages.std().item():.4f}")
    # breakpoint()

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None


@register_advantage("stepwise_gae")
def compute_stepwise_gae_advantages(
    rewards: torch.Tensor,
    stepwise_values: torch.Tensor,
    num_denoise_steps: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Step-wise GAE advantages for Trace-VLA using λ-return.

    This implements denoise-step-level advantage estimation where each denoising
    step is treated as an action in an internal MDP. Uses TD(λ) to propagate
    the chunk reward backwards to all denoising steps.

    Key insight: In the denoising MDP, only the final step receives the real
    chunk reward. Without λ-return, intermediate steps would only learn from
    bootstrap estimates (V_{t+1} - V_t), which provides weak learning signal.
    By using λ-return, the chunk reward is propagated to all steps.

    The TD(λ) return at step t is:
        G_t = (1-λ) * V_{t+1} + λ * G_{t+1}   (for t < T-1)
        G_{T-1} = R_chunk                      (terminal step)

    The advantage at step t is:
        A_t = G_t - V_t

    Args:
        rewards (torch.Tensor): Chunk-level rewards. Shape: [B] or [B, 1]
        stepwise_values (torch.Tensor): Per-step value estimates. Shape: [B, num_denoise_steps]
        num_denoise_steps (int): Number of denoising steps.
        gamma (float): Discount factor for denoising MDP (typically 1.0).
        gae_lambda (float): Lambda for TD(λ) return. 1.0 = MC return (all steps get
            the same chunk reward), 0.0 = TD(0) (only last step uses chunk reward).
            Default 0.95 provides good balance.
        normalize_advantages (bool): Whether to normalize advantages.
        loss_mask (torch.Tensor, optional): Mask for valid samples. Shape: [B] or [B, num_denoise_steps]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - advantages: Shape [B, num_denoise_steps]
            - returns: Shape [B, num_denoise_steps]
    """
    # Handle reward shape
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)  # [B] -> [B, 1]

    B, T = stepwise_values.shape
    device = stepwise_values.device
    dtype = stepwise_values.dtype

    advantages = torch.zeros(B, T, dtype=dtype, device=device)
    returns = torch.zeros(B, T, dtype=dtype, device=device)

    # Squeeze chunk reward: [B, 1] -> [B]
    chunk_reward = rewards.squeeze(-1)

    # ========== λ-return backward propagation ==========
    # This propagates the chunk reward signal to ALL denoising steps,
    # not just the last one. This is critical for learning.

    # Last step: directly uses chunk reward
    returns[:, -1] = chunk_reward
    advantages[:, -1] = chunk_reward - stepwise_values[:, -1]

    # Backward pass: propagate reward signal through λ-return
    for t in reversed(range(T - 1)):
        # TD(λ) return: G_t = γ * [(1-λ) * V_{t+1} + λ * G_{t+1}]
        # When λ=1: G_t = γ * G_{t+1} (pure MC, all steps get chunk_reward)
        # When λ=0: G_t = γ * V_{t+1} (pure TD(0), intermediate steps bootstrap)
        # When γ=1 (typical for denoising MDP): G_t = (1-λ)*V_{t+1} + λ*G_{t+1}
        returns[:, t] = gamma * (
            (1 - gae_lambda) * stepwise_values[:, t + 1]
            + gae_lambda * returns[:, t + 1]
        )
        advantages[:, t] = returns[:, t] - stepwise_values[:, t]

    # Handle loss mask
    if loss_mask is not None:
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(-1).expand(-1, T)  # [B] -> [B, T]

    # Normalize advantages
    if normalize_advantages:
        if loss_mask is not None:
            valid_adv = advantages[loss_mask.bool()]
        else:
            valid_adv = advantages.flatten()

        if valid_adv.numel() > 0:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std().clamp(min=1e-8)
            advantages = (advantages - adv_mean) / adv_std

    return advantages, returns


@register_advantage("hierarchical_gae")
@torch.no_grad()
def compute_hierarchical_gae(
    rewards: torch.Tensor,
    chunk_values: torch.Tensor,
    stepwise_values: torch.Tensor,
    num_denoise_steps: int,
    chunk_gamma: float = 0.99,
    step_gamma: float = 1.0,
    gae_lambda: float = 0.95,
    step_gae_lambda: Optional[float] = None,  # If None, uses gae_lambda for step-level
    normalize_advantages: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[dict, dict]:
    """
    Compute Hierarchical GAE for HUA-RL (Hierarchical Uncertainty-Aware RL).

    This implements bi-level advantage estimation:
    1. Chunk-Level GAE: Standard GAE across chunks for long-term credit assignment
    2. Step-Level λ-return: Within-chunk credit assignment with chunk-level bootstrap

    The key innovation is that step-level returns bootstrap to the NEXT chunk's
    value estimate, enabling proper credit propagation across both time scales:

        Step-level target = R_chunk + γ_chunk * V_chunk(o_{t+1})

    This allows each denoising step to learn its contribution not just to the
    current chunk's reward, but to the entire episode's return.

    Mathematical formulation:
        Chunk-Level GAE:
            δ_c^t = R_chunk^t + γ_chunk * V_c(o_{t+1}) - V_c(o_t)
            A_c^t = Σ_{l=0}^{∞} (γ_chunk * λ)^l * δ_c^{t+l}

        Step-Level λ-return with bootstrap:
            Terminal_target = R_chunk^t + γ_chunk * V_c(o_{t+1})
            G_s^{t,τ} = γ_step * [(1-λ) * V_s^{t,τ+1} + λ * G_s^{t,τ+1}]
            A_s^{t,τ} = G_s^{t,τ} - V_s^{t,τ}

    Args:
        rewards: [T, B] or [T, B, 1] - Chunk-level rewards (T = num_chunks, same format as baseline)
        chunk_values: [T+1, B] or [T+1, B, 1] - Chunk-level values (includes bootstrap)
        stepwise_values: [T, B, num_denoise_steps] - Step-level values
        num_denoise_steps: Number of denoising steps per chunk
        chunk_gamma: Discount factor for chunk-level transitions (default 0.99)
        step_gamma: Discount factor for step-level transitions (default 1.0)
        gae_lambda: Lambda for chunk-level GAE (default 0.95)
        step_gae_lambda: Lambda for step-level λ-return (default None, uses gae_lambda).
            Set to 1.0 for pure MC (all steps use chunk_returns as target).
        normalize_advantages: Whether to normalize advantages
        loss_mask: [T, B] - Mask for valid chunks
        dones: [T+1, B] or [T+1, B, 1] - Episode done flags

    Returns:
        Tuple of two dicts:
            chunk_results: {
                "advantages": [T, B] or [T, B, 1] - Chunk-level advantages (same shape as rewards)
                "returns": [T, B] or [T, B, 1] - Chunk-level returns
            }
            step_results: {
                "advantages": [T, B, num_denoise_steps] - Step-level advantages
                "returns": [T, B, num_denoise_steps] - Step-level returns
            }
    """
    # === HIERARCHICAL GAE: 函数入口断点 ===
    # print("[HIERARCHICAL GAE] 进入 compute_hierarchical_gae 函数")
    # breakpoint()

    # Detach inputs to ensure no gradient tracking
    # This prevents gradient graph contamination and memory explosion
    rewards = rewards.detach()
    chunk_values = chunk_values.detach()
    stepwise_values = stepwise_values.detach()
    if dones is not None:
        dones = dones.detach()

    # Handle input shapes - [T, B] or [T, B, 1] format (same as baseline!)
    # T = num_chunks (time dimension first, like baseline)
    num_chunks = rewards.shape[0]
    B = rewards.shape[1]
    T = num_denoise_steps
    device = rewards.device
    dtype = rewards.dtype

    # Handle dones - ensure correct shape and type
    # dones shape: [T+1, B] or [T+1, B, 1]
    if dones is None:
        dones = torch.zeros(num_chunks + 1, B, dtype=torch.bool, device=device)
    else:
        # Convert to bool if needed (dones might be float 0/1)
        if dones.dtype != torch.bool:
            dones = dones.bool()

    # ===== Phase 1: Chunk-Level GAE (like baseline) =====
    # Squeeze rewards to [T, B] to match baseline exactly
    rewards_2d = rewards.squeeze(-1) if rewards.dim() == 3 else rewards  # [T, B]
    chunk_values_2d = chunk_values.squeeze(-1) if chunk_values.dim() == 3 else chunk_values  # [T+1, B]
    dones_2d = dones.squeeze(-1) if dones.dim() == 3 else dones  # [T+1, B]

    chunk_advantages = torch.zeros_like(rewards_2d)  # [T, B]
    chunk_returns = torch.zeros_like(rewards_2d)  # [T, B]

    # === HIERARCHICAL GAE BREAKPOINT 1: 输入 ===
    # print(f"\n[HIERARCHICAL GAE] Input shapes: rewards={rewards.shape}, chunk_values={chunk_values.shape}, stepwise_values={stepwise_values.shape}")
    # print(f"[HIERARCHICAL GAE] dones={dones.shape}, chunk_gamma={chunk_gamma}, gae_lambda={gae_lambda}, step_gae_lambda={step_gae_lambda}")
    # print(f"[HIERARCHICAL GAE] num_chunks={num_chunks}, B={B}, T={T}")
    # breakpoint()

    gae = 0  # Scalar, will broadcast (like baseline)
    for t in reversed(range(num_chunks)):
        # TD error at chunk level - same indexing as baseline: [t] instead of [:, t]
        not_done = (~dones_2d[t + 1]).float()  # [B]

        delta = rewards_2d[t] + chunk_gamma * chunk_values_2d[t + 1] * not_done - chunk_values_2d[t]

        # GAE accumulation (same as baseline)
        gae = delta + chunk_gamma * gae_lambda * not_done * gae
        chunk_advantages[t] = gae
        chunk_returns[t] = gae + chunk_values_2d[t]

        # === HIERARCHICAL GAE BREAKPOINT 2: 第一次迭代 (t=num_chunks-1) ===
        # if t == num_chunks - 1:
        #     print(f"[HIERARCHICAL GAE] Phase1 first iter (t={t}): delta.shape={delta.shape}, delta[:3]={delta.flatten()[:3].tolist()}")
        #     print(f"[HIERARCHICAL GAE] not_done[:3]={not_done.flatten()[:3].tolist()}")
        #     print(f"[HIERARCHICAL GAE] chunk_advantages[{t}][:3]={chunk_advantages[t].flatten()[:3].tolist()}")
        #     breakpoint()

    # === HIERARCHICAL GAE BREAKPOINT 3: Phase 1 输出 ===
    # print(f"\n[HIERARCHICAL GAE] Phase1 done: chunk_advantages.shape={chunk_advantages.shape}")
    # print(f"[HIERARCHICAL GAE] chunk_advantages[:3]={chunk_advantages.flatten()[:3].tolist()}")
    # print(f"[HIERARCHICAL GAE] chunk_advantages.mean()={chunk_advantages.mean().item():.4f}, std={chunk_advantages.std().item():.4f}")
    # breakpoint()

    # ===== Phase 2: Step-Level λ-return with chunk bootstrap =====
    # stepwise_values: [T, B, num_denoise_steps]
    # step_advantages/returns: [T, B, num_denoise_steps]
    step_advantages = torch.zeros(num_chunks, B, T, dtype=dtype, device=device)
    step_returns = torch.zeros(num_chunks, B, T, dtype=dtype, device=device)

    # Use step_gae_lambda for step-level (defaults to gae_lambda if not specified)
    _step_gae_lambda = step_gae_lambda if step_gae_lambda is not None else gae_lambda

    for chunk_idx in range(num_chunks):
        # Terminal target for this chunk's denoising MDP:
        # R_chunk + γ_chunk * V_chunk(next)
        # Use 2D variables (already squeezed) for consistency
        not_done = (~dones_2d[chunk_idx + 1]).float()  # [B]
        terminal_target = rewards_2d[chunk_idx] + chunk_gamma * chunk_values_2d[chunk_idx + 1] * not_done  # [B]

        # Backward λ-return within chunk
        # Last step targets the terminal
        step_returns[chunk_idx, :, -1] = terminal_target
        step_advantages[chunk_idx, :, -1] = terminal_target - stepwise_values[chunk_idx, :, -1]

        # Propagate backwards through denoising steps
        for tau in reversed(range(T - 1)):
            # λ-return: G_τ = γ * [(1-λ) * V_{τ+1} + λ * G_{τ+1}]
            step_returns[chunk_idx, :, tau] = step_gamma * (
                (1 - _step_gae_lambda) * stepwise_values[chunk_idx, :, tau + 1]
                + _step_gae_lambda * step_returns[chunk_idx, :, tau + 1]
            )
            step_advantages[chunk_idx, :, tau] = (
                step_returns[chunk_idx, :, tau] - stepwise_values[chunk_idx, :, tau]
            )

    # ===== Normalization =====
    if normalize_advantages:
        # Normalize chunk advantages
        chunk_valid = chunk_advantages.flatten()
        step_valid = step_advantages.flatten()

        if chunk_valid.numel() > 0:
            chunk_mean = chunk_valid.mean()
            chunk_std = chunk_valid.std().clamp(min=1e-8)
            chunk_advantages = (chunk_advantages - chunk_mean) / chunk_std

        if step_valid.numel() > 0:
            step_mean = step_valid.mean()
            step_std = step_valid.std().clamp(min=1e-8)
            step_advantages = (step_advantages - step_mean) / step_std

    # === HIERARCHICAL GAE BREAKPOINT 4: 最终输出 (归一化后) ===
    # print(f"\n[HIERARCHICAL GAE] Final output (post-norm):")
    # print(f"[HIERARCHICAL GAE] chunk_advantages.shape={chunk_advantages.shape}, chunk_advantages[:3]={chunk_advantages.flatten()[:3].tolist()}")
    # print(f"[HIERARCHICAL GAE] chunk_advantages.mean()={chunk_advantages.mean().item():.4f}, std={chunk_advantages.std().item():.4f}")
    # print(f"[HIERARCHICAL GAE] step_advantages.shape={step_advantages.shape}")
    # breakpoint()

    chunk_results = {
        "advantages": chunk_advantages,
        "returns": chunk_returns,
    }
    step_results = {
        "advantages": step_advantages,
        "returns": step_returns,
    }

    return chunk_results, step_results


# =============================================================================
# FlowRL: Optimal Transport Credit Assignment (OTCA)
# =============================================================================


@register_advantage("flowrl_otca")
def compute_flowrl_otca_advantages(
    rewards: torch.Tensor,
    stepwise_values: torch.Tensor,
    velocities: torch.Tensor,
    num_denoise_steps: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    otca_mode: str = "multiply",
    min_transport_weight: float = 0.1,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute FlowRL OTCA (Optimal Transport Credit Assignment) advantages.

    This implements credit assignment based on Benamou-Brenier optimal transport:
        W_2²(p_noise, p_action) = ∫₀¹ ||v(x,t)||² dt

    Each denoising step's "transport cost" ||v(x_t,t)||² determines its credit:
    - High velocity norm = large probability mass movement = high credit
    - Low velocity norm = fine-tuning = lower credit

    This is theoretically grounded in OT theory, unlike heuristic timestep weighting.

    Args:
        rewards: [B] or [B, 1] - Chunk-level rewards
        stepwise_values: [B, T] - Per-step value estimates
        velocities: [B, T, action_horizon, action_dim] - Velocity predictions at each step
        num_denoise_steps: Number of denoising steps T
        gamma: Discount factor within denoising MDP
        gae_lambda: Lambda for TD(λ) return
        normalize_advantages: Whether to normalize final advantages
        loss_mask: [B] or [B, T] - Sample validity mask
        otca_mode: How to apply OT weights:
            - "multiply": A_otca = transport_weight * A_step
            - "weighted_sum": A_otca = Σ(transport_weight * A_step) / Σ(transport_weight)
            - "target_weight": Use weights to reweight the return targets
        min_transport_weight: Minimum weight to prevent zero weights

    Returns:
        Tuple of:
            - advantages: [B, T] - OTCA-weighted advantages
            - returns: [B, T] - Step-level returns
            - metadata: dict with transport_weights and other info
    """
    # Handle reward shape
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)  # [B] -> [B, 1]

    B, T = stepwise_values.shape
    device = stepwise_values.device
    dtype = stepwise_values.dtype

    # ===== Step 1: Compute OT-based transport weights =====
    # ||v||² at each step: [B, T, H, D] -> [B, T]
    velocity_norm_sq = (velocities ** 2).sum(dim=(-1, -2))  # [B, T]

    # Normalize to get credit proportion
    total_transport = velocity_norm_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    transport_weights = velocity_norm_sq / total_transport * T  # [B, T], sums to T

    # Apply clamp to prevent extreme weights that amplify advantages
    transport_weights = transport_weights.clamp(min=0.7, max=1.3)

    # ===== Step 2: Compute base λ-return =====
    chunk_reward = rewards.squeeze(-1)  # [B]
    returns = torch.zeros(B, T, dtype=dtype, device=device)
    advantages = torch.zeros(B, T, dtype=dtype, device=device)

    # Last step gets chunk reward
    returns[:, -1] = chunk_reward
    advantages[:, -1] = chunk_reward - stepwise_values[:, -1]

    # Backward propagation
    for t in reversed(range(T - 1)):
        returns[:, t] = gamma * (
            (1 - gae_lambda) * stepwise_values[:, t + 1]
            + gae_lambda * returns[:, t + 1]
        )
        advantages[:, t] = returns[:, t] - stepwise_values[:, t]

    # ===== Step 3: Normalize BEFORE weighting =====
    if normalize_advantages:
        if loss_mask is not None:
            if loss_mask.dim() == 1:
                loss_mask = loss_mask.unsqueeze(-1).expand(-1, T)
            valid_adv = advantages[loss_mask.bool()]
        else:
            valid_adv = advantages.flatten()

        if valid_adv.numel() > 0:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std().clamp(min=1e-8)
            advantages = (advantages - adv_mean) / adv_std

    # ===== Step 4: Apply OT weights AFTER normalization =====
    if otca_mode == "multiply":
        # Direct multiplication: steps with high transport get amplified
        weighted_advantages = advantages * transport_weights

    elif otca_mode == "weighted_sum":
        # Weighted average (for aggregation across steps)
        # Here we keep per-step form for PPO, but could be used differently
        weight_sum = transport_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_advantages = advantages * transport_weights / weight_sum * T

    elif otca_mode == "target_weight":
        # Use weights to modify the return targets before computing advantages
        # This affects the value targets during training
        weighted_returns = returns * transport_weights
        weighted_advantages = weighted_returns - stepwise_values * transport_weights

    else:
        raise ValueError(f"Unknown otca_mode: {otca_mode}")

    # Metadata for logging
    metadata = {
        "transport_weights": transport_weights,  # [B, T]
        "transport_weight_mean": transport_weights.mean(),
        "transport_weight_std": transport_weights.std(),
        "velocity_norm_mean": velocity_norm_sq.sqrt().mean(),
    }

    return weighted_advantages, returns, metadata


@register_advantage("flowrl_hierarchical_otca")
def compute_flowrl_hierarchical_otca_advantages(
    rewards: torch.Tensor,
    chunk_values: torch.Tensor,
    stepwise_values: torch.Tensor,
    velocities: torch.Tensor,
    num_denoise_steps: int,
    chunk_gamma: float = 0.99,
    step_gamma: float = 1.0,
    gae_lambda: float = 0.95,
    step_gae_lambda: Optional[float] = None,  # Defaults to gae_lambda if not specified
    normalize_advantages: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    consistency_scores: Optional[torch.Tensor] = None,
    consistency_mode: str = "multiply",
    otca_mode: str = "multiply",
    min_transport_weight: float = 0.1,
    **kwargs,
) -> tuple[dict, dict, dict]:
    """
    Compute FlowRL Hierarchical OTCA advantages with optional consistency weighting.

    This is the complete FlowRL advantage computation that combines:
    1. Chunk-level GAE for long-term credit assignment
    2. Step-level λ-return with chunk bootstrap
    3. Optimal Transport credit weighting based on ||v||²
    4. Trajectory consistency weighting (optional)

    Mathematical formulation:
        Transport credit: w_t^OT = ||v(x_t, t)||² / Σ_τ ||v(x_τ, τ)||²
        Consistency: c = 1 / (1 + Var(x0_predictions))
        Final weight: w_t = c * w_t^OT (if consistency enabled)

    Args:
        rewards: [B, num_chunks] - Chunk-level rewards
        chunk_values: [B, num_chunks + 1] - Chunk-level value estimates
        stepwise_values: [B, num_chunks, T] - Step-level value estimates
        velocities: [B, num_chunks, T, action_horizon, action_dim] - Velocity predictions
        num_denoise_steps: T
        chunk_gamma: Discount for chunk-level
        step_gamma: Discount for step-level
        gae_lambda: GAE lambda
        normalize_advantages: Whether to normalize
        loss_mask: [B, num_chunks] - Mask
        dones: [B, num_chunks + 1] - Episode done flags
        consistency_scores: [B, num_chunks] - Pre-computed trajectory consistency (optional)
        consistency_mode: How to apply consistency ("multiply", "gate", "none")
        otca_mode: How to apply OT weights
        min_transport_weight: Minimum transport weight

    Returns:
        Tuple of:
            - chunk_results: dict with chunk advantages and returns
            - step_results: dict with step advantages, returns, and weights
            - metadata: dict with transport weights, consistency, etc.
    """
    # Handle input shapes
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)
        chunk_values = chunk_values.unsqueeze(0)
        stepwise_values = stepwise_values.unsqueeze(0)
        velocities = velocities.unsqueeze(0)
        if consistency_scores is not None:
            consistency_scores = consistency_scores.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, num_chunks = rewards.shape
    T = num_denoise_steps
    device = rewards.device
    dtype = rewards.dtype

    # Handle dones
    if dones is None:
        dones = torch.zeros(B, num_chunks + 1, dtype=torch.bool, device=device)

    # ===== Phase 1: Compute OT transport weights =====
    # velocities: [B, num_chunks, T, H, D]
    velocity_norm_sq = (velocities ** 2).sum(dim=(-1, -2))  # [B, num_chunks, T]
    total_transport = velocity_norm_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, num_chunks, 1]
    transport_weights = velocity_norm_sq / total_transport * T  # [B, num_chunks, T]
    # Apply clamp to prevent extreme weights that amplify advantages
    transport_weights = transport_weights.clamp(min=0.7, max=1.3)

    # ===== Phase 2: Apply consistency weighting if provided =====
    if consistency_scores is not None and consistency_mode != "none":
        # consistency_scores: [B, num_chunks]
        if consistency_mode == "multiply":
            # Scale transport weights by consistency
            consistency_expanded = consistency_scores.unsqueeze(-1)  # [B, num_chunks, 1]
            combined_weights = transport_weights * consistency_expanded
        elif consistency_mode == "gate":
            # Gate: only apply transport weights for consistent trajectories
            consistency_expanded = consistency_scores.unsqueeze(-1)
            consistency_gate = (consistency_expanded > 0.5).float()
            combined_weights = transport_weights * consistency_gate + (1 - consistency_gate) * torch.ones_like(transport_weights)
        else:
            combined_weights = transport_weights
    else:
        combined_weights = transport_weights

    # ===== Phase 3: Chunk-level GAE =====
    chunk_advantages = torch.zeros(B, num_chunks, dtype=dtype, device=device)
    chunk_returns = torch.zeros(B, num_chunks, dtype=dtype, device=device)

    gae = torch.zeros(B, dtype=dtype, device=device)
    for t in reversed(range(num_chunks)):
        not_done = (~dones[:, t + 1]).float()
        delta = rewards[:, t] + chunk_gamma * chunk_values[:, t + 1] * not_done - chunk_values[:, t]
        gae = delta + chunk_gamma * gae_lambda * not_done * gae
        chunk_advantages[:, t] = gae
        chunk_returns[:, t] = gae + chunk_values[:, t]

    # ===== Phase 4: Step-level λ-return with OTCA weighting =====
    step_advantages = torch.zeros(B, num_chunks, T, dtype=dtype, device=device)
    step_returns = torch.zeros(B, num_chunks, T, dtype=dtype, device=device)

    # Use step_gae_lambda for step-level (defaults to gae_lambda if not specified)
    _step_gae_lambda = step_gae_lambda if step_gae_lambda is not None else gae_lambda

    for chunk_idx in range(num_chunks):
        not_done = (~dones[:, chunk_idx + 1]).float()
        terminal_target = rewards[:, chunk_idx] + chunk_gamma * chunk_values[:, chunk_idx + 1] * not_done

        # Backward λ-return
        step_returns[:, chunk_idx, -1] = terminal_target
        step_advantages[:, chunk_idx, -1] = terminal_target - stepwise_values[:, chunk_idx, -1]

        for tau in reversed(range(T - 1)):
            step_returns[:, chunk_idx, tau] = step_gamma * (
                (1 - _step_gae_lambda) * stepwise_values[:, chunk_idx, tau + 1]
                + _step_gae_lambda * step_returns[:, chunk_idx, tau + 1]
            )
            step_advantages[:, chunk_idx, tau] = (
                step_returns[:, chunk_idx, tau] - stepwise_values[:, chunk_idx, tau]
            )

    # ===== Phase 5: Normalization (BEFORE weighting to prevent advantage amplification) =====
    if normalize_advantages:
        # Normalize chunk advantages
        if loss_mask is not None:
            if loss_mask.dim() == 1:
                loss_mask = loss_mask.unsqueeze(0).expand(B, -1)
            chunk_valid = chunk_advantages[loss_mask.bool()]
            step_mask = loss_mask.unsqueeze(-1).expand(-1, -1, T)
            step_valid = step_advantages[step_mask.bool()]
        else:
            chunk_valid = chunk_advantages.flatten()
            step_valid = step_advantages.flatten()

        if chunk_valid.numel() > 0:
            chunk_mean = chunk_valid.mean()
            chunk_std = chunk_valid.std().clamp(min=1e-8)
            chunk_advantages = (chunk_advantages - chunk_mean) / chunk_std

        if step_valid.numel() > 0:
            step_mean = step_valid.mean()
            step_std = step_valid.std().clamp(min=1e-8)
            step_advantages = (step_advantages - step_mean) / step_std

    # Apply OTCA weights AFTER normalization
    if otca_mode == "multiply":
        weighted_step_advantages = step_advantages * combined_weights
    elif otca_mode == "weighted_sum":
        weight_sum = combined_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_step_advantages = step_advantages * combined_weights / weight_sum * T
    else:
        weighted_step_advantages = step_advantages * combined_weights

    # Squeeze if needed
    if squeeze_output:
        chunk_advantages = chunk_advantages.squeeze(0)
        chunk_returns = chunk_returns.squeeze(0)
        weighted_step_advantages = weighted_step_advantages.squeeze(0)
        step_returns = step_returns.squeeze(0)
        combined_weights = combined_weights.squeeze(0)
        transport_weights = transport_weights.squeeze(0)

    chunk_results = {
        "advantages": chunk_advantages,
        "returns": chunk_returns,
    }
    step_results = {
        "advantages": weighted_step_advantages,
        "returns": step_returns,
        "weights": combined_weights,
    }
    metadata = {
        "transport_weights": transport_weights,
        "combined_weights": combined_weights,
        "transport_weight_mean": transport_weights.mean() if not squeeze_output else transport_weights.mean(),
        "transport_weight_std": transport_weights.std() if not squeeze_output else transport_weights.std(),
    }
    if consistency_scores is not None:
        metadata["consistency_scores"] = consistency_scores
        metadata["consistency_mean"] = consistency_scores.mean()

    return chunk_results, step_results, metadata

