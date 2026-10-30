"""Implementation imported by plot_attention.py; no Isaac Sim is required."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot saved AME attention in scanner xy order.")
    parser.add_argument("attention_file", type=Path)
    parser.add_argument("--grid_shape", type=int, nargs=2, default=(21, 33), metavar=("ROWS", "COLS"))
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    weights = np.load(args.attention_file)
    if weights.ndim == 4:
        weights = weights[:, :, None, :, :]
    if weights.ndim != 5:
        raise ValueError(f"Expected [step, env, (head), query, point], got {weights.shape}")
    rows, cols = args.grid_shape
    points = weights.shape[-1]
    if points == ((rows + 1) // 2) * ((cols + 1) // 2):
        rows, cols = (rows + 1) // 2, (cols + 1) // 2
    elif points != rows * cols:
        raise ValueError(f"{points} attention points do not match grid {args.grid_shape}")
    out = args.output_dir or args.attention_file.parent / "attn_vis"
    out.mkdir(parents=True, exist_ok=True)
    for step in range(min(len(weights), args.max_frames)):
        heads = weights.shape[2]
        fig, axes = plt.subplots(1, heads, figsize=(6 * heads, 4), squeeze=False)
        for head in range(heads):
            ax = axes[0, head]
            terrain_attention = weights[step, 0, head, 0].reshape(rows, cols)
            chart = ax.imshow(terrain_attention, origin="lower", extent=(-0.8, 0.8, -0.5, 0.5), cmap="viridis")
            ax.set(xlabel="Forward x (m)", ylabel="Lateral y (m)", title=f"Step {step}, head {head}")
            fig.colorbar(chart, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"attention_{step:04d}.png", dpi=120)
        plt.close(fig)
    print(f"Attention figures: {out}")


if __name__ == "__main__":
    main()
