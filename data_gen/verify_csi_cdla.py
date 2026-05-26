"""Verify generated CSI dataset: shapes, stats, and 3-sample heatmaps."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_gen/csi_cdla')
    parser.add_argument('--out',      default='data_gen/csi_cdla/verify.png')
    args = parser.parse_args()

    train = np.load(os.path.join(args.data_dir, 'train_data.npy'))
    test  = np.load(os.path.join(args.data_dir, 'test_data.npy'))
    train_paths = np.load(os.path.join(args.data_dir, 'train_paths.npy'))

    # Shape checks
    assert train.shape == (70000, 2, 64, 64), f'bad train shape: {train.shape}'
    assert test.shape  == (10000, 2, 64, 64), f'bad test shape: {test.shape}'
    assert train.dtype == np.float32
    print('Shapes OK')
    print(f'train n_paths: min={train_paths.min()} max={train_paths.max()} '
          f'mean={train_paths.mean():.1f}')

    # 3-sample visualisation: magnitude heatmap
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(train), 3, replace=False)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for row, idx in enumerate(idxs):
        x = train[idx]           # [2, 64, 64]
        mag = np.sqrt(x[0]**2 + x[1]**2)  # [64, 64]
        ax = axes[row]
        im = ax.imshow(mag, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Azimuth bin')
        ax.set_ylabel('Delay tap')
        ax.set_title(f'Sample {idx}  |  n_paths={train_paths[idx]}')
        plt.colorbar(im, ax=ax, label='|H|')

    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    print(f'Saved visualisation to {args.out}')


if __name__ == '__main__':
    main()
