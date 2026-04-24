"""Plot comparison of env/episode_len from tensorboard logs."""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_scalar(log_dir: str, tag: str):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth_ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    result = np.zeros_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def rolling_uncertainty(values: np.ndarray, window: int = 20):
    half_w = window // 2
    lower = np.zeros_like(values)
    upper = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        chunk = values[start:end]
        mu = np.mean(chunk)
        sigma = np.std(chunk)
        lower[i] = mu - sigma
        upper[i] = mu + sigma
    return lower, upper


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
        },
        {
            "name": "GRPO",
            "log_dir": f"{log_base}/20260410-18:09:48-maniskill_grpo_openpi_pi05/tensorboard",
            "color": "#27ae60",
            "band_color": "#27ae60",
        },
    ]

    tag = "env/episode_len"

    fig, ax = plt.subplots(figsize=(9, 6))

    for run in runs:
        try:
            steps, raw_values = read_tensorboard_scalar(run["log_dir"], tag)
        except Exception:
            print(f"Warning: {run['name']} has no {tag} data, skipping")
            continue

        # Truncate at 400 steps
        mask = steps <= 400
        steps = steps[mask]
        raw_values = raw_values[mask]

        smoothed = smooth_ema(raw_values, alpha=0.08)
        lower, upper = rolling_uncertainty(raw_values, window=30)

        ax.fill_between(
            steps, lower, upper,
            alpha=0.15, color=run["band_color"], linewidth=0,
            label=f"{run['name']} (±1σ band)"
        )
        ax.plot(steps, smoothed, color=run["color"], linewidth=2.0,
                label=run["name"])

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Episode Length", fontsize=13)
    ax.set_xlim(left=0)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/scripts/traj_len_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
