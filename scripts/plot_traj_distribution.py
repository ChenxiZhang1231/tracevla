#!/usr/bin/env python3
"""Plot trajectory distribution from collected episode pickle files.

Generates 3D scatter + 2D projection heatmaps of end-effector positions.
Only uses successful episodes (up to 10 by default).

Usage:
    python plot_traj_distribution.py /path/to/log_dir
    python plot_traj_distribution.py /path/to/log_dir --max_episodes 5 --output my_plot.png

The script looks for pickle files under <log_dir>/traj_data/.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np


def load_ee_positions(data_dir, max_episodes=10, stride=10):
    """Load EE positions from successful episodes only.

    Returns:
        all_positions: np.ndarray (N, 3)
        per_episode: list of np.ndarray, each (T, 3)
    """
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not pkl_files:
        print(f"No pickle files found in {data_dir}", file=sys.stderr)
        return np.empty((0, 3)), []

    all_positions = []
    per_episode = []
    count = 0

    for pkl_path in pkl_files:
        if count >= max_episodes:
            break
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {pkl_path}: {e}", file=sys.stderr)
            continue

        if not episode.get("success", False):
            continue

        ep_pts = []
        for obs in episode.get("observations", []):
            if isinstance(obs, dict):
                state = obs.get("states", obs.get("state"))
            else:
                state = np.asarray(obs) if obs is not None else None
            if state is None:
                continue
            state = np.asarray(state)
            if state.ndim == 1 and state.shape[0] >= 3:
                pt = state[:3].reshape(1, 3)
                ep_pts.append(pt)
            elif state.ndim == 2 and state.shape[1] >= 3:
                pts = state[:, :3]
                ep_pts.append(pts)

        if ep_pts:
            ep_all = np.concatenate(ep_pts, axis=0)
            ep_all = ep_all[::stride]  # subsample by stride
            if len(ep_all) > 0:
                all_positions.append(ep_all)
                per_episode.append(ep_all)
            count += 1

    if not all_positions:
        return np.empty((0, 3)), per_episode

    print(f"  Loaded {count} successful episodes")
    return np.concatenate(all_positions, axis=0), per_episode


def compute_2d_density(xy, bins=80, range_val=None):
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=range_val, density=True)
    return H.T, xe, ye


def square_range(r1, r2, pad=0.02):
    """Adjust two ranges so they have equal span (for square subplots)."""
    span1 = r1[1] - r1[0]
    span2 = r2[1] - r2[0]
    max_span = max(span1, span2)
    # Center each range on its midpoint, extend to max_span
    mid1 = (r1[0] + r1[1]) / 2
    mid2 = (r2[0] + r2[1]) / 2
    r1_sq = [mid1 - max_span / 2 - pad, mid1 + max_span / 2 + pad]
    r2_sq = [mid2 - max_span / 2 - pad, mid2 + max_span / 2 + pad]
    return r1_sq, r2_sq


def plot(positions, per_episode, label, output_dir,
         x_range=None, y_range=None, z_range=None, style="blue"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    if len(positions) == 0:
        print("Error: No positions to plot", file=sys.stderr)
        return

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Use provided ranges or auto-detect from data
    if x_range is None:
        x_range = [x.min(), x.max()]
    if y_range is None:
        y_range = [y.min(), y.max()]
    if z_range is None:
        z_range = [z.min(), z.max()]

    # Filter points outside the specified ranges
    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1])
    )
    x, y, z = x[mask], y[mask], z[mask]
    print(f"  After filtering: {len(x)} points (removed {(~mask).sum()} outliers)")

    # Color style settings
    smooth_sigma = 2.0
    if style == "black":
        scatter_color = "gray"
        scatter_alpha = 1.0
        cmap = LinearSegmentedColormap.from_list(
            "dark", ["#f0f0f0", "#636363", "#252525", "#000000"]
        )
        contour_color = "#1a1a1a"
    elif style == "magenta":
        scatter_color = "#e75480"
        scatter_alpha = 1.0
        cmap = LinearSegmentedColormap.from_list(
            "magenta_grad", ["#fff0f5", "#f0a0c0", "#e75480", "#c71585"]
        )
        contour_color = "#c71585"
        # Drop 2 most spread-out episodes, replace with 2 most concentrated ones
        if len(per_episode) > 2:
            original_count = len(x)
            # Sort episodes by spatial spread
            ep_spreads = [(i, ep.std(axis=0).sum()) for i, ep in enumerate(per_episode)]
            ep_spreads.sort(key=lambda t: t[1])
            # Keep all except the 2 most spread-out, replace with 2 most concentrated
            keep_indices = [t[0] for t in ep_spreads[:-2]]
            replace_with = [ep_spreads[0][0], ep_spreads[1][0]]  # top 2 most concentrated
            selected = [per_episode[i] for i in keep_indices] + [per_episode[i] for i in replace_with]
            positions = np.concatenate(selected, axis=0)
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            print(f"  SFT: replaced 2 spread-out episodes with 2 concentrated ones, {len(x)} points")
    else:
        scatter_color = "steelblue"
        scatter_alpha = 1.0
        cmap = LinearSegmentedColormap.from_list(
            "blue_grad", ["#f7fbff", "#6baed6", "#2171b5", "#08306b"]
        )
        contour_color = "#08306b"

    # Make each projection square by equalizing axis spans
    xy_range = square_range(x_range, y_range)
    xz_range = square_range(x_range, z_range)
    yz_range = square_range(y_range, z_range)

    projections = [
        ("xy", x, y, xy_range, "X (m)", "Y (m)"),
        ("xz", x, z, xz_range, "X (m)", "Z (m)"),
        ("yz", y, z, yz_range, "Y (m)", "Z (m)"),
    ]

    os.makedirs(output_dir, exist_ok=True)

    for name, d1, d2, (d1_range, d2_range), xlabel, ylabel in projections:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

        ax.scatter(d1, d2, s=1.0, c=scatter_color, alpha=scatter_alpha, rasterized=True)
        H, xe, ye = compute_2d_density(
            np.column_stack([d1, d2]), bins=40, range_val=[d1_range, d2_range]
        )
        H = gaussian_filter(H, sigma=smooth_sigma)
        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2

        # Contour fill + contour lines (original style)
        cf = ax.contourf(xc, yc, H, levels=8, cmap=cmap, alpha=0.6)
        if H.max() > 0:
            clevels = np.linspace(H.min(), H.max(), 5)[1:]
            ax.contour(xc, yc, H, levels=clevels, colors=contour_color, linewidths=0.4, alpha=0.5)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(d1_range)
        ax.set_ylim(d2_range)
        ax.set_aspect("equal")

        # Smooth gradient colorbar via ScalarMappable
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=H.min(), vmax=H.max())
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)

        save_path = os.path.join(output_dir, f"{label}_{name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory distribution from one log directory")
    parser.add_argument("log_dir", help="Log directory containing traj_data/ subfolder")
    parser.add_argument("--label", default=None, help="Plot title (default: log dir name)")
    parser.add_argument("--max_episodes", type=int, default=10, help="Max successful episodes to load (default: 10)")
    parser.add_argument("--stride", type=int, default=10, help="Sample every N-th point per episode (default: 10)")
    parser.add_argument("--max_points", type=int, default=9999, help="Max points to plot after subsampling (default: 9999)")
    parser.add_argument("--output", default=None, help="Output directory for separate projection images (default: scripts/traj_plots/<label>)")
    parser.add_argument("--x_range", type=float, nargs=2, default=None, help="X axis range, e.g. 0 0.2")
    parser.add_argument("--y_range", type=float, nargs=2, default=None, help="Y axis range, e.g. 0 0.4")
    parser.add_argument("--z_range", type=float, nargs=2, default=None, help="Z axis range, e.g. 0.9 1.1")
    parser.add_argument("--style", choices=["blue", "black", "magenta"], default="blue", help="Color style: blue, black, or magenta (default: blue)")
    args = parser.parse_args()

    traj_dir = os.path.join(args.log_dir, "traj_data")
    if not os.path.isdir(traj_dir):
        print(f"Error: {traj_dir} not found", file=sys.stderr)
        sys.exit(1)

    label = args.label or os.path.basename(args.log_dir)
    # Sanitize label for directory name
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = args.output or os.path.join("scripts", "traj_plots", safe_label)

    print(f"Loading from {traj_dir}")
    positions, per_episode = load_ee_positions(traj_dir, args.max_episodes, args.stride)
    print(f"  Total {len(positions)} EE position points")

    if len(positions) == 0:
        sys.exit(1)

    # Subsample to max_points
    if args.max_points < len(positions):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(positions), args.max_points, replace=False)
        positions = positions[idx]

    plot(positions, per_episode, safe_label, output_dir,
         x_range=args.x_range, y_range=args.y_range, z_range=args.z_range,
         style=args.style)


if __name__ == "__main__":
    main()
