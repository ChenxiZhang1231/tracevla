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

"""
Video Progress Reward Model for Dense Reward in VLA RL.

This module implements a reward model that predicts the progress of a task
based on the VLM's prefix output. The progress is a value in [0, 1] indicating
how far along the current observation is in a successful trajectory.

Key features:
- Input: VLM prefix_output (frozen, stable embeddings)
- Output: progress ∈ [0, 1]
- Training: Online learning from successful trajectories during RL
- Usage: All trajectories use progress difference as dense reward
"""

import random
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoProgressRewardModel(nn.Module):
    """
    Progress Prediction Reward Model (only uses prefix).

    Parameters: ~1.3M
    Input: prefix_output [B, 968, 2048] (for Pi0.5) or [B, 816, 1024] (for Pi0)
    Output: progress [B] ∈ [0, 1]

    Key: Uses mask to filter invalid tokens (padding images)
    """

    def __init__(
        self,
        prefix_dim: int = 2048,
        num_images_in_input: int = 1,
        lang_token_len: int = 200,
        total_image_slots: int = 3,
        compress_dim: int = 512,
        hidden_sizes: tuple = (512, 256, 128),
        activation: str = "relu",
    ):
        """
        Args:
            prefix_dim: Dimension of prefix tokens (2048 for Pi0.5, 1024 for Pi0)
            num_images_in_input: Number of actual images (1 for ManiSkill)
            lang_token_len: Number of language tokens (200 for Pi0.5, 48 for Pi0)
            total_image_slots: Total image slots in VLM (3 for both Pi0 and Pi0.5)
            compress_dim: Dimension to compress prefix to
            hidden_sizes: Hidden layer sizes for progress MLP
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()

        self.prefix_dim = prefix_dim
        self.num_images_in_input = num_images_in_input
        self.lang_token_len = lang_token_len

        # Create mask: only pool valid tokens
        # Valid tokens = 256 * num_images + lang_token_len
        # For Pi0.5 with 1 image: 256 + 200 = 456 valid tokens out of 968
        tokens_per_image = 256
        prefix_mask = (
            [True] * tokens_per_image * num_images_in_input  # Valid image tokens
            + [False] * tokens_per_image * (total_image_slots - num_images_in_input)  # Padding images (exclude)
            + [True] * lang_token_len  # Language tokens
        )
        self.register_buffer("prefix_mask", torch.tensor(prefix_mask))

        # Activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Prefix compression: prefix_dim → compress_dim (~1.05M parameters)
        self.prefix_compress = nn.Sequential(
            nn.Linear(prefix_dim, compress_dim),
            nn.LayerNorm(compress_dim),
            act_fn(),
        )

        # Progress MLP: compress_dim → ... → 1 (~0.25M parameters)
        layers = []
        in_dim = compress_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                act_fn(),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.progress_head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for module in [self.prefix_compress, self.progress_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Initialize final layer with small weights for stable initial predictions
        final_linear = self.progress_head[-2]  # Last linear before sigmoid
        if isinstance(final_linear, nn.Linear):
            nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def forward(self, prefix_output: torch.Tensor) -> torch.Tensor:
        """
        Predict progress from prefix output.

        Args:
            prefix_output: [B, seq_len, prefix_dim] - VLM prefix output
                          For Pi0.5: [B, 968, 2048]
                          For Pi0: [B, 816, 1024]

        Returns:
            progress: [B] ∈ [0, 1]
        """
        # 1. Filter with mask, only take valid tokens
        valid_output = prefix_output[:, self.prefix_mask, :]  # [B, num_valid, prefix_dim]

        # 2. Mean pool valid tokens
        pooled = valid_output.mean(dim=1)  # [B, prefix_dim]

        # 3. Compress and predict
        prefix_feat = self.prefix_compress(pooled)  # [B, compress_dim]
        progress = self.progress_head(prefix_feat).squeeze(-1)  # [B]

        return progress


class OnlineProgressRewardTrainer:
    """
    Online trainer for the Progress Reward Model during RL.

    This trainer:
    1. Computes progress rewards during rollout (no gradient)
    2. Stores successful trajectories in a buffer
    3. Updates the reward model from successful trajectories (with gradient)
    """

    def __init__(
        self,
        reward_model: VideoProgressRewardModel,
        optimizer: torch.optim.Optimizer,
        buffer_size: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            reward_model: The progress reward model to train
            optimizer: Optimizer for the reward model
            buffer_size: Maximum number of samples in the success buffer
            device: Device for training
        """
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.buffer_size = buffer_size
        self.device = device or next(reward_model.parameters()).device

        # Buffer for successful trajectory data
        # Each entry: {"prefix_output": Tensor, "progress": float}
        self.success_buffer = deque(maxlen=buffer_size)

        # Track statistics
        self.total_successes = 0
        self.total_updates = 0

    def compute_progress_reward(
        self,
        prefix_output: torch.Tensor,
        prev_progress: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute progress reward during rollout (no gradient).

        Args:
            prefix_output: [B, seq_len, prefix_dim] - VLM output (frozen)
            prev_progress: [B] - Previous step's progress prediction

        Returns:
            progress_reward: [B] - Reward based on progress difference
            progress: [B] - Current progress prediction
            prefix_output_detached: [B, seq_len, prefix_dim] - Detached prefix for later training
        """
        with torch.no_grad():
            progress = self.reward_model(prefix_output)

        progress_reward = progress - prev_progress

        # Return detached prefix for potential later training
        return progress_reward, progress, prefix_output.detach()

    def add_success_trajectory(
        self,
        trajectory_data: list[dict],
    ):
        """
        Add a successful trajectory to the buffer.

        Args:
            trajectory_data: List of dicts, each containing:
                - "prefix_output": [seq_len, prefix_dim] - Detached VLM output
        """
        T = len(trajectory_data)
        if T == 0:
            return

        self.total_successes += 1

        for t, step in enumerate(trajectory_data):
            progress_label = t / (T - 1) if T > 1 else 1.0
            self.success_buffer.append({
                "prefix_output": step["prefix_output"].cpu(),  # Store on CPU to save GPU memory
                "progress": progress_label,
            })

    def train_reward_model(
        self,
        batch_size: int = 64,
        num_updates: int = 1,
    ) -> Optional[float]:
        """
        Update the reward model using successful trajectories.

        Args:
            batch_size: Batch size for training
            num_updates: Number of gradient updates

        Returns:
            Average loss, or None if not enough data
        """
        if len(self.success_buffer) < batch_size:
            return None

        self.reward_model.train()
        total_loss = 0.0

        for update_idx in range(num_updates):
            # Sample batch
            indices = random.sample(range(len(self.success_buffer)), batch_size)
            batch = [self.success_buffer[i] for i in indices]

            # Debug: check individual sample shapes
            if update_idx == 0:
                sample_shapes = [b["prefix_output"].shape for b in batch[:3]]
                print(f"[ProgressRM Train] Sample shapes (first 3): {sample_shapes}")

            prefix_output = torch.stack([b["prefix_output"] for b in batch]).to(self.device)

            # Match dtype to reward model's dtype
            model_dtype = next(self.reward_model.parameters()).dtype
            prefix_output = prefix_output.to(dtype=model_dtype)

            # Debug: check stacked shape
            if update_idx == 0:
                print(f"[ProgressRM Train] Stacked prefix_output shape: {prefix_output.shape}, dtype: {prefix_output.dtype}, "
                      f"expected: [batch, seq_len, dim] = [{batch_size}, 968, 2048] or similar")

            progress_label = torch.tensor(
                [b["progress"] for b in batch],
                device=self.device,
                dtype=torch.float32,
            )

            # Forward
            pred_progress = self.reward_model(prefix_output)

            # Loss - compute in float32 for numerical stability
            loss = F.mse_loss(pred_progress.float(), progress_label)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.total_updates += 1

        self.reward_model.eval()
        return total_loss / num_updates

    def warmup(
        self,
        demo_trajectories: list[list[dict]],
        warmup_steps: int = 100,
        batch_size: int = 64,
    ) -> list[float]:
        """
        Pre-train the reward model before RL starts.

        Args:
            demo_trajectories: List of successful demo trajectories.
                Each trajectory is a list of dicts with "prefix_output".
            warmup_steps: Number of training steps
            batch_size: Batch size for training

        Returns:
            List of loss values during warmup
        """
        # Build warmup buffer
        warmup_buffer = []
        for demo in demo_trajectories:
            T = len(demo)
            for t, step in enumerate(demo):
                warmup_buffer.append({
                    "prefix_output": step["prefix_output"],
                    "progress": t / (T - 1) if T > 1 else 1.0,
                })

        if len(warmup_buffer) < batch_size:
            print(f"[RM Warmup] Warning: Not enough warmup data ({len(warmup_buffer)} < {batch_size})")
            batch_size = max(1, len(warmup_buffer))

        self.reward_model.train()
        losses = []

        for step in range(warmup_steps):
            # Sample batch
            batch = random.sample(warmup_buffer, min(batch_size, len(warmup_buffer)))

            prefix_output = torch.stack([b["prefix_output"] for b in batch]).to(self.device)
            progress_label = torch.tensor(
                [b["progress"] for b in batch],
                device=self.device,
                dtype=torch.float32,
            )

            # Forward
            pred_progress = self.reward_model(prefix_output)

            # Loss - compute in float32 for numerical stability
            loss = F.mse_loss(pred_progress.float(), progress_label)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if step % 20 == 0:
                print(f"[RM Warmup] Step {step}, Loss: {loss.item():.4f}")

        self.reward_model.eval()
        return losses

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "buffer_size": len(self.success_buffer),
            "total_successes": self.total_successes,
            "total_updates": self.total_updates,
        }

    def save_checkpoint(self, path: str):
        """Save reward model and buffer to checkpoint."""
        torch.save({
            "model_state_dict": self.reward_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "buffer": list(self.success_buffer),
            "total_successes": self.total_successes,
            "total_updates": self.total_updates,
        }, path)

    def load_checkpoint(self, path: str):
        """Load reward model and buffer from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.reward_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.success_buffer = deque(checkpoint["buffer"], maxlen=self.buffer_size)
        self.total_successes = checkpoint["total_successes"]
        self.total_updates = checkpoint["total_updates"]


def create_progress_reward_model(
    model_type: str = "pi05",
    num_images_in_input: int = 1,
    compress_dim: int = 512,
    hidden_sizes: tuple = (512, 256, 128),
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
) -> tuple[VideoProgressRewardModel, OnlineProgressRewardTrainer]:
    """
    Factory function to create a progress reward model and its trainer.

    Args:
        model_type: "pi05" or "pi0"
        num_images_in_input: Number of images in input
        compress_dim: Compression dimension
        hidden_sizes: Hidden layer sizes
        learning_rate: Learning rate for optimizer
        device: Device for the model

    Returns:
        Tuple of (reward_model, trainer)
    """
    if model_type == "pi05":
        prefix_dim = 2048
        lang_token_len = 200
    elif model_type == "pi0":
        prefix_dim = 1024
        lang_token_len = 48
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    reward_model = VideoProgressRewardModel(
        prefix_dim=prefix_dim,
        num_images_in_input=num_images_in_input,
        lang_token_len=lang_token_len,
        compress_dim=compress_dim,
        hidden_sizes=hidden_sizes,
    )

    if device:
        reward_model = reward_model.to(device)

    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    trainer = OnlineProgressRewardTrainer(
        reward_model=reward_model,
        optimizer=optimizer,
        device=device,
    )

    return reward_model, trainer
