#!/usr/bin/env python3
"""Plot trajectory distribution with leave-2-out: drop episodes (i, i+1) for i in 0..8."""

import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


def compute_2d_density(xy, bins=80, range_val=None):
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=range_val, density=True)
    return H.T, xe, ye


def square_range(r1, r2, pad=0.02):
    span1 = r1[1] - r1[0]
    span2 = r2[1] - r2[0]
    max_span = max(span1, span2)
    mid1 = (r1[0] + r1[1]) / 2
    mid2 = (r2[0] + r2[1]) / 2
    return [mid1 - max_span/2 - pad, mid1 + max_span/2 + pad], \
           [mid2 - max_span/2 - pad, mid2 + max_span/2 + pad]


def load_episodes(data_dir, selected_indices, stride=10):
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    all_pts = []
    for i in selected_indices:
        with open(pkl_files[i], "rb") as f:
            ep = pickle.load(f)
        if not ep.get("success", False):
            continue
        pts = []
        for obs in ep.get("observations", []):
            s = obs.get("states", obs.get("state")) if isinstance(obs, dict) else np.asarray(obs)
            if s is None:
                continue
            s = np.asarray(s)
            if s.ndim == 1 and s.shape[0] >= 3:
                pts.append(s[:3].reshape(1, 3))
            elif s.ndim == 2 and s.shape[1] >= 3:
                pts.append(s[:, :3])
        if pts:
            all_pts.append(np.concatenate(pts, axis=0)[::stride])
    return np.concatenate(all_pts, axis=0) if all_pts else np.empty((0, 3))


def main():
    data_dir = "logs/20260423-12:29:15/traj_data"
    base_dir = "scripts/traj_plots/CrossAttention"

    # Settings
    cmap = LinearSegmentedColormap.from_list("blue_grad", ["#f7fbff", "#6baed6", "#2171b5", "#08306b"])
    contour_color = "#08306b"
    x_range, y_range, z_range = [-0.05, 0.2], [-0.05, 0.3], [0.9, 1.15]

    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    n = len(pkl_files)
    print(f"Total episodes: {n}")

    for drop_idx in range(n - 1):
        keep = [i for i in range(n) if i not in (drop_idx, drop_idx + 1)]
        positions = load_episodes(data_dir, keep, stride=10)

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        mask = (x >= x_range[0]) & (x <= x_range[1]) & \
               (y >= y_range[0]) & (y <= y_range[1]) & \
               (z >= z_range[0]) & (z <= z_range[1])
        x, y, z = x[mask], y[mask], z[mask]

        folder = f"drop_{drop_idx+1}_{drop_idx+2}"
        output_dir = os.path.join(base_dir, folder)
        os.makedirs(output_dir, exist_ok=True)

        xy_r = square_range(x_range, y_range)
        xz_r = square_range(x_range, z_range)
        yz_r = square_range(y_range, z_range)

        projections = [
            ("xy", x, y, xy_r, "X (m)", "Y (m)"),
            ("xz", x, z, xz_r, "X (m)", "Z (m)"),
            ("yz", y, z, yz_r, "Y (m)", "Z (m)"),
        ]

        for name, d1, d2, (d1r, d2r), xl, yl in projections:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
            ax.scatter(d1, d2, s=1.0, c="steelblue", alpha=1.0, rasterized=True)
            H, xe, ye = compute_2d_density(np.column_stack([d1, d2]), bins=40, range_val=[d1r, d2r])
            H = gaussian_filter(H, sigma=2.0)
            xc = (xe[:-1] + xe[1:]) / 2
            yc = (ye[:-1] + ye[1:]) / 2
            cf = ax.contourf(xc, yc, H, levels=8, cmap=cmap, alpha=0.6)
            if H.max() > 0:
                cl = np.linspace(H.min(), H.max(), 5)[1:]
                ax.contour(xc, yc, H, levels=cl, colors=contour_color, linewidths=0.4, alpha=0.5)
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
            ax.set_xlim(d1r)
            ax.set_ylim(d2r)
            ax.set_aspect("equal")
            sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(H.min(), H.max()))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.8)
            fig.savefig(os.path.join(output_dir, f"CrossAttention_{name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"drop ep {drop_idx+1}&{drop_idx+2}: {len(x)} pts -> {output_dir}")

    print("Done! 9 folders saved.")


if __name__ == "__main__":
    main()
