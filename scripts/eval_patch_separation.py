"""
Patch Separation Score (PSS) and AUROC for LatentMIM CSI representations.

Ground truth: a patch is labelled 'path' if its peak noiseless amplitude
is within -20 dB of the sample peak (same threshold as n_paths counting).

Metrics (averaged over test samples):
  PSS   = ||mu_path - mu_noise||^2 / (var_path + var_noise)
  AUROC = area under ROC using distance-to-noise-centroid as score
"""

import numpy as np
from sklearn.metrics import roc_auc_score

PATCH_SIZE      = 2
PATCH_GRID_H    = 20
PATCH_GRID_W    = 32
PATH_THRESH_DB  = -20.0

def get_patch_labels(clean_sample):
    """
    clean_sample: [64, 64] complex64 (full noiseless channel)
    Returns bool array [PATCH_GRID_H, PATCH_GRID_W]
    """
    mag = np.abs(clean_sample[:40, :])   # crop to signal region [40, 64]
    peak = mag.max()
    if peak == 0:
        return np.zeros((PATCH_GRID_H, PATCH_GRID_W), dtype=bool)
    thresh = peak * 10 ** (PATH_THRESH_DB / 20)   # amplitude threshold
    labels = np.zeros((PATCH_GRID_H, PATCH_GRID_W), dtype=bool)
    for i in range(PATCH_GRID_H):
        for j in range(PATCH_GRID_W):
            patch = mag[i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                        j*PATCH_SIZE:(j+1)*PATCH_SIZE]
            labels[i, j] = patch.max() >= thresh
    return labels


def compute_metrics(reps, clean):
    """
    reps:  [N, 640, 384]
    clean: [N, 64, 64] complex64
    """
    N = len(reps)
    pss_list, auroc_list = [], []

    for n in range(N):
        r = reps[n]                          # [640, 384]
        lab = get_patch_labels(clean[n]).flatten()   # [640] bool

        n_path  = lab.sum()
        n_noise = (~lab).sum()
        if n_path == 0 or n_noise == 0:
            continue

        r_path  = r[lab]     # [n_path, 384]
        r_noise = r[~lab]    # [n_noise, 384]

        mu_p = r_path.mean(0)
        mu_n = r_noise.mean(0)

        var_p = ((r_path  - mu_p) ** 2).sum(1).mean()
        var_n = ((r_noise - mu_n) ** 2).sum(1).mean()

        pss = np.dot(mu_p - mu_n, mu_p - mu_n) / (var_p + var_n + 1e-8)
        pss_list.append(pss)

        # AUROC: score = distance to noise centroid (larger = more path-like)
        scores = np.linalg.norm(r - mu_n, axis=1)
        auroc  = roc_auc_score(lab.astype(int), scores)
        auroc_list.append(auroc)

        if (n + 1) % 1000 == 0:
            print(f'  {n+1}/{N}  PSS={np.mean(pss_list):.3f}  AUROC={np.mean(auroc_list):.3f}')

    return np.array(pss_list), np.array(auroc_list)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps',  default='scripts/representations.npy')
    parser.add_argument('--clean', default='data_gen/csi_cdla/test_clean.npy')
    args = parser.parse_args()

    print('Loading representations and clean channel data...')
    reps  = np.load(args.reps)    # [N, 640, 384]
    clean = np.load(args.clean)   # [N, 64, 64] complex64

    print(f'Evaluating {len(reps)} test samples...')
    pss, auroc = compute_metrics(reps, clean)

    print(f'\n--- Results ---')
    print(f'PSS   mean={pss.mean():.3f}  std={pss.std():.3f}  median={np.median(pss):.3f}')
    print(f'AUROC mean={auroc.mean():.3f}  std={auroc.std():.3f}  median={np.median(auroc):.3f}')
    print(f'(AUROC: 0.5=random, 1.0=perfect)')


if __name__ == '__main__':
    main()
