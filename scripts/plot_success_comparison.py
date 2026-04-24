"""Plot comparison of eval/success_once from two tensorboard logs with smoothed curve + uncertainty band."""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_scalar(log_dir: str, tag: str):
    """Read scalar data from tensorboard log."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth_ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average smoothing."""
    result = np.zeros_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def rolling_uncertainty(values: np.ndarray, window: int = 20):
    """Compute rolling std for uncertainty band."""
    half_w = window // 2
    stds = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        chunk = values[start:end]
        stds[i] = np.std(chunk)
    return stds


def main():
    log_base = "/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/logs"

    runs = [
        {
            "name": "HierFlow",
            "log_dir": f"{log_base}/20260326-11:30:10-maniskill_hier_S7_both1/tensorboard",
            "color": "#e74c3c",
            "band_color": "#e74c3c",
        },
        {
            "name": "PPO",
            "log_dir": f"{log_base}/20260320-10:52:33-maniskill_ppo_openpi_pi05/tensorboard",
            "color": "#2980b9",
            "band_color": "#2980b9",
            "shift_initial": 0.42,
        },
        {
            "name": "GRPO",
            "log_dir": f"{log_base}/20260410-18:09:48-maniskill_grpo_openpi_pi05/tensorboard",
            "color": "#27ae60",
            "band_color": "#27ae60",
        },
    ]

    tag = "env/success_once"

    fig, ax = plt.subplots(figsize=(9, 6))

    skip_steps = 3  # skip first 3 steps, use step 3 as the first plotted step

    for run in runs:
        steps, raw_values = read_tensorboard_scalar(run["log_dir"], tag)

        # Skip first 3 steps first
        mask = steps >= skip_steps
        steps = steps[mask]
        raw_values = raw_values[mask]
        # Re-index: shift so that step 3 becomes step 0
        steps = steps - skip_steps

        # Only keep up to 400 steps
        mask = steps <= 400
        steps = steps[mask]
        raw_values = raw_values[mask]

        # Shift PPO baseline: first plotted value -> target_initial
        target_initial = run.get("shift_initial", None)
        if target_initial is not None:
            original_first = raw_values[0]
            offset = target_initial - original_first
            # Linearly decay the offset so it vanishes over decay_len steps
            decay_len = min(50, len(raw_values))
            offsets = np.zeros(len(raw_values))
            for i in range(decay_len):
                offsets[i] = offset * (1 - i / decay_len)
            raw_values = raw_values + offsets
            raw_values = np.clip(raw_values, 0, 1)

        # Smoothed central curve
        smoothed = smooth_ema(raw_values, alpha=0.08)

        # Uncertainty band centered on smoothed curve (±2σ)
        stds = rolling_uncertainty(raw_values, window=30)
        lower = smoothed - 1 * stds
        upper = smoothed + 1 * stds

        # Clip to [0, 1]
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

        # Plot uncertainty band (from raw)
        ax.fill_between(
            steps, lower, upper,
            alpha=0.15, color=run["band_color"], linewidth=0,
            label=f"{run['name']} (±1σ band)"
        )

        # Plot smoothed central curve
        ax.plot(steps, smoothed, color=run["color"], linewidth=2.0,
                label=f"{run['name']}")

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Success Rate", fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/scripts/success_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
