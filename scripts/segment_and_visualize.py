# scripts/segment_and_visualize.py
"""
For 5 random test samples:
  1. Run AgglomerativeClustering (cosine+average) on [640, 384] patch representations
  2. Overlay cluster-coloured patch boundaries on the original CSI image
  3. Run t-SNE on 256 patch vectors
  4. Plot row: [CSI |H| dB (clean) | CSI with patch overlay | t-SNE]

Usage:
  python scripts/segment_and_visualize.py \
    --reps   scripts/representations.npy \
    --data   data_gen/csi_cdla/test_data.npy \
    --paths  data_gen/csi_cdla/test_paths.npy \
    --out    scripts/segmentation.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

N_SAMPLES  = 5
PATCH_GRID_H = 20     # delay rows  (40 taps / patch_size 2)
PATCH_GRID_W = 32     # azimuth cols (64 bins  / patch_size 2)
PATCH_SIZE   = 2      # pixels per patch side
K_MIN, K_MAX = 2, 5
OVERLAY_ALPHA = 0.45  # transparency of the cluster fill

# Fixed high-contrast palette (up to 5 clusters).
# Chosen to pop against viridis (dark-blue → yellow) background.
PALETTE = [
    (0.95, 0.10, 0.10),   # vivid red
    (0.10, 0.60, 1.00),   # electric blue
    (0.10, 0.90, 0.30),   # bright green
    (1.00, 0.55, 0.00),   # orange
    (0.85, 0.10, 0.95),   # magenta
]


def cluster_sample(reps, n_paths):
    k = int(np.clip(n_paths, K_MIN, K_MAX))
    labels = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(reps)
    return labels, k


def run_tsne(reps):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(reps)


def draw_patch_overlay(ax, img_2d, labels, k):
    """Show img_2d with semi-transparent cluster-coloured rectangles + grid lines."""
    ax.imshow(img_2d, aspect='auto', origin='lower', cmap='viridis')
    for i in range(PATCH_GRID_H):        # array row (i=0 → bottom of display)
        for j in range(PATCH_GRID_W):    # array col  (j=0 → left of display)
            label = labels[i * PATCH_GRID_W + j]
            color = PALETTE[label % len(PALETTE)]
            rect = mpatches.Rectangle(
                (j * PATCH_SIZE - 0.5, i * PATCH_SIZE - 0.5),
                PATCH_SIZE, PATCH_SIZE,
                linewidth=0.4,
                edgecolor='white',
                facecolor=color,
                alpha=OVERLAY_ALPHA,
            )
            ax.add_patch(rect)


def k_sweep(reps, test_data, idx, ks=(2, 3, 4, 5), out='scripts/segmentation_ksweep.png'):
    """One sample image with overlay repeated for each k value."""
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4))
    rep = reps[idx]
    x   = test_data[idx]
    for ax, k in zip(axes, ks):
        labels = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(rep)
        draw_patch_overlay(ax, x[0], labels, k)
        ax.set_title(f'k = {k}')
        ax.set_xlabel('Azimuth bin')
        ax.set_ylabel('Delay tap')
    fig.suptitle(f'Sample index {idx} — varying k', y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved to {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps',       default='scripts/representations.npy')
    parser.add_argument('--data',       default='data_gen/csi_cdla/test_data.npy')
    parser.add_argument('--paths',      default='data_gen/csi_cdla/test_paths.npy')
    parser.add_argument('--out',        default='scripts/segmentation.png')
    parser.add_argument('--ksweep-out', default='scripts/segmentation_ksweep.png')
    parser.add_argument('--seed',       type=int, default=0)
    args = parser.parse_args()

    reps      = np.load(args.reps)    # [10000, 256, 192]
    test_data = np.load(args.data)    # [10000, 1, 64, 64]  dB magnitude
    n_paths   = np.load(args.paths)   # [10000]

    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(len(reps), N_SAMPLES, replace=False)

    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(12, 4 * N_SAMPLES))

    for row, idx in enumerate(idxs):
        rep   = reps[idx]
        x     = test_data[idx]
        k_val = int(n_paths[idx])

        labels, k = cluster_sample(rep, k_val)
        tsne_xy   = run_tsne(rep)
        colours   = [PALETTE[c % len(PALETTE)] for c in labels]

        # Col 0: clean CSI image
        ax0 = axes[row, 0]
        ax0.imshow(x[0], aspect='auto', origin='lower', cmap='viridis')
        ax0.set_xlabel('Azimuth bin')
        ax0.set_ylabel('Delay tap')
        ax0.set_title(f'CSI |H| dB   n_paths={k_val}')

        # Col 1: same image with patch-boundary overlay
        ax1 = axes[row, 1]
        draw_patch_overlay(ax1, x[0], labels, k)
        ax1.set_xlabel('Azimuth bin')
        ax1.set_ylabel('Delay tap')
        ax1.set_title(f'Patch clusters  k={k}')

        # Col 2: t-SNE
        ax2 = axes[row, 2]
        ax2.scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=colours, s=8, alpha=0.8)
        ax2.set_title('t-SNE  (colour = cluster)')
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f'Saved to {args.out}')

    reps      = np.load(args.reps)
    test_data = np.load(args.data)
    k_sweep(reps, test_data, idx=int(idxs[0]), out=args.ksweep_out)


if __name__ == '__main__':
    main()
