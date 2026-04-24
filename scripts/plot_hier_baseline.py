"""Plot Hierarchical S7 success curve with sampled points, baselines, and improvement arrow."""

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


def main():
    log_dir = "/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/logs/20260326-11:30:10-maniskill_hier_S7_both1/tensorboard"
    tag = "env/success_once"

    steps, values = read_tensorboard_scalar(log_dir, tag)

    # Skip first 3 steps
    skip = 3
    mask = steps >= skip
    steps = steps[mask] - skip
    values = values[mask]

    # Sample every k steps, always include the max point
    k = 5
    max_idx = np.argmax(values)
    # Only keep data up to the max point
    steps = steps[:max_idx + 1]
    values = values[:max_idx + 1]
    sampled_idx = np.arange(0, len(steps), k)
    if max_idx not in sampled_idx:
        sampled_idx = np.sort(np.append(sampled_idx, max_idx))
    s_steps = steps[sampled_idx]
    s_values = values[sampled_idx]

    # Convert to percentage
    s_pct = s_values * 100
    final_pct = values.max() * 100

    # Baselines (percentage)
    baselines = [
        {"value": 40.2, "label": "Few-Shot", "color": "#FF8C00"},
        {"value": 70.7, "label": "Few-Shot+GRPO", "color": "#FF8C00"},
        {"value": 82.7, "label": "Few-Shot+PPO", "color": "#FF8C00"},
    ]

    # Improvement arrow: from Few-Shot to final
    low_base = 40.2
    abs_improve = final_pct - low_base
    rel_improve = (final_pct - low_base) / low_base * 100

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 7))

    # Horizontal baseline dashed lines
    label_offsets = {"Few-Shot": 1.5, "Few-Shot+GRPO": 1.5, "Few-Shot+PPO": 3.0}
    for bl in baselines:
        ax.axhline(y=bl["value"], color=bl["color"], linestyle="--", linewidth=1.2, alpha=0.8)
        offset = label_offsets.get(bl["label"], 1.5)
        ax.text(
            s_steps[-1] * 0.6, bl["value"] - offset,
            f'{bl["label"]}: {bl["value"]:.1f}%',
            fontsize=14, color=bl["color"], va="top", fontweight="bold",
        )

    # Final result reference line (blue dashed)
    ax.axhline(y=final_pct, color="#2980b9", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        s_steps[-1] * 0.6, final_pct - 1.5,
        f"HierFlow: {final_pct:.1f}%",
        fontsize=14, color="#2980b9", va="top", fontweight="bold",
    )

    # Sampled curve with markers
    ax.plot(
        s_steps, s_pct,
        color="#2c3e50", linewidth=1.8, alpha=0.9, zorder=3,
    )
    ax.scatter(
        s_steps, s_pct,
        s=18, color="#2c3e50", marker="s", zorder=4, edgecolors="white", linewidths=0.3,
    )

    # Red vertical improvement arrow on the left (two segments with gap for text)
    arrow_x = -15
    mid_y = (low_base + final_pct) / 2
    text_gap = 3.5  # half-height of text gap

    # Upper segment: mid+gap -> final_pct (with arrowhead)
    ax.annotate(
        "",
        xy=(arrow_x, final_pct),
        xytext=(arrow_x, mid_y + text_gap),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#e74c3c",
            lw=2.5,
            mutation_scale=18,
        ),
    )
    # Lower segment: low_base -> mid-gap (no arrowhead)
    ax.annotate(
        "",
        xy=(arrow_x, mid_y - text_gap),
        xytext=(arrow_x, low_base),
        arrowprops=dict(
            arrowstyle="-",
            color="#e74c3c",
            lw=2.5,
        ),
    )

    # Arrow label (centered on the gap)
    ax.text(
        arrow_x, mid_y,
        f"+{abs_improve:.1f}\n(↑{rel_improve:.1f}%)",
        fontsize=11, color="#e74c3c", fontweight="bold",
        ha="center", va="center",
    )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title("HierFlow", fontsize=14)
    ax.set_xlim(left=-50, right=s_steps[-1] + 80)
    ax.set_ylim(30, 100)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = "/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/scripts/hier_baseline_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
