# scripts/segment_and_visualize.py
"""
For 5 random test samples:
  1. Run AgglomerativeClustering on [256, 384] patch representations
  2. Upsample 16x16 label map -> 64x64 (nearest neighbour)
  3. Run t-SNE on 256 patch vectors
  4. Plot row: [CSI magnitude | segmentation map | t-SNE], annotated with n_paths

Usage:
  python scripts/segment_and_visualize.py \
    --reps   scripts/representations.npy \
    --data   data_gen/csi_cdla/test_data.npy \
    --paths  data_gen/csi_cdla/test_paths.npy \
    --out    scripts/segmentation_results.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.ndimage import zoom

N_SAMPLES  = 5
PATCH_GRID = 16
K_MIN, K_MAX = 2, 5


def cluster_sample(reps_256_384, n_paths):
    k = int(np.clip(n_paths, K_MIN, K_MAX))
    labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(reps_256_384)
    return labels, k


def upsample_labels(labels_256, patch_grid=PATCH_GRID, target=64):
    label_map = labels_256.reshape(patch_grid, patch_grid)
    scale = target / patch_grid
    return zoom(label_map.astype(float), scale, order=0).astype(int)


def run_tsne(reps_256_384):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(reps_256_384)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps',  default='scripts/representations.npy')
    parser.add_argument('--data',  default='data_gen/csi_cdla/test_data.npy')
    parser.add_argument('--paths', default='data_gen/csi_cdla/test_paths.npy')
    parser.add_argument('--out',   default='scripts/segmentation_results.png')
    parser.add_argument('--seed',  type=int, default=0)
    args = parser.parse_args()

    reps      = np.load(args.reps)    # [10000, 256, 384]
    test_data = np.load(args.data)    # [10000, 2, 64, 64]
    n_paths   = np.load(args.paths)   # [10000]

    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(len(reps), N_SAMPLES, replace=False)

    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(12, 4 * N_SAMPLES))

    for row, idx in enumerate(idxs):
        rep   = reps[idx]
        x     = test_data[idx]
        k_val = int(n_paths[idx])

        labels, k = cluster_sample(rep, k_val)
        label_map = upsample_labels(labels)
        tsne_xy   = run_tsne(rep)
        colours   = [cmap(c / max(k - 1, 1)) for c in labels]

        # Plot 1: CSI magnitude
        mag = np.sqrt(x[0] ** 2 + x[1] ** 2)
        ax0 = axes[row, 0]
        ax0.imshow(mag, aspect='auto', origin='lower', cmap='viridis')
        ax0.set_xlabel('Azimuth bin')
        ax0.set_ylabel('Delay tap')
        ax0.set_title(f'CSI |H|   n_paths={k_val}')

        # Plot 2: Segmentation map
        ax1 = axes[row, 1]
        seg_rgb = np.array([[cmap(c / max(k - 1, 1))[:3]
                              for c in row_vals]
                             for row_vals in label_map])
        ax1.imshow(seg_rgb, aspect='auto', origin='lower')
        ax1.set_xlabel('Azimuth bin')
        ax1.set_ylabel('Delay tap')
        ax1.set_title(f'Segmentation  k={k}')

        # Plot 3: t-SNE
        ax2 = axes[row, 2]
        ax2.scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=colours, s=8, alpha=0.8)
        ax2.set_title('t-SNE  (colour = cluster)')
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
