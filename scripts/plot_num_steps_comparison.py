"""Plot success rate comparison by num_steps from tensorboard logs."""

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
            "name": "num_steps=4",
            "log_dir": f"{log_base}/20260326-11:30:10-maniskill_hier_S7_both1/tensorboard",
            "color": "#e74c3c",
            "band_color": "#e74c3c",
            "shift_initial": 0.4358,
        },
        {
            "name": "num_steps=5",
            "log_dir": f"{log_base}/20260326-10:40:12-maniskill_hier_S12_steps12/tensorboard",
            "color": "#2980b9",
            "band_color": "#2980b9",
        },
        {
            "name": "num_steps=6",
            "log_dir": f"{log_base}/20260325-11:38:17-maniskill_hier_S6_both_10/tensorboard",
            "color": "#27ae60",
            "band_color": "#27ae60",
            "shift_initial": 0.4358,
        },
    ]

    tag = "env/success_once"
    fig, ax = plt.subplots(figsize=(9, 6))

    for run in runs:
        steps, raw_values = read_tensorboard_scalar(run["log_dir"], tag)

        # Shift initial value to target
        target_initial = run.get("shift_initial", None)
        if target_initial is not None:
            original_first = raw_values[0]
            offset = target_initial - original_first
            decay_len = min(50, len(raw_values))
            offsets = np.zeros(len(raw_values))
            for i in range(decay_len):
                offsets[i] = offset * (1 - i / decay_len)
            raw_values = raw_values + offsets
            raw_values = np.clip(raw_values, 0, 1)

        # Truncate at 300 steps
        mask = steps <= 300
        steps = steps[mask]
        raw_values = raw_values[mask]

        smoothed = smooth_ema(raw_values, alpha=0.08)
        stds = rolling_uncertainty(raw_values, window=30)
        lower = smoothed - 1 * stds
        upper = smoothed + 1 * stds

        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

        ax.fill_between(
            steps, lower, upper,
            alpha=0.15, color=run["band_color"], linewidth=0,
        )
        ax.plot(steps, smoothed, color=run["color"], linewidth=2.0,
                label=run["name"])

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Success Rate", fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/scripts/num_steps_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
