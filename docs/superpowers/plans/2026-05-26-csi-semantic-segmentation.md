# CSI Semantic Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-step pipeline that generates CDL-A CSI data, trains LatentMIM on it, and produces semantic segmentation + t-SNE visualisations confirming that latent patch representations capture propagation path structure.

**Architecture:** Sionna CDL-A generates per-path CIR; 1D FFT on the antenna dimension and fixed delay-grid discretisation produce [2, 64, 64] float32 tensors (real/imag channels); LatentMIM (ViT-Small, patch_size=4, grid_size=16) is trained with a 2-channel patch embedding; post-training, patch representations from the online encoder are clustered with AgglomerativeClustering and visualised alongside t-SNE plots, both colour-coded by the same cluster labels.

**Tech Stack:** Python 3.8+, TensorFlow/Sionna 1.2.0, PyTorch 2.1, scikit-learn, matplotlib, numpy

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `data_gen/generate_csi_cdla.py` | Sionna CDL-A → [2, 64, 64] delay-azimuth tensors + path counts |
| Create | `data_gen/verify_csi_cdla.py` | Shape checks + 3-sample heatmap visualisation |
| Modify | `datasets.py` | Add `CSICDLADataset` class and `csi_cdla()` function |
| Modify | `models_lmim.py` | Add `in_chans` parameter to `LMIM.__init__` and `build_lmim` |
| Create | `configs/lmim_csi.yaml` | CSI-specific Hydra training config |
| Modify | `main_lmim.py` | Skip ImageNet transforms when `dataset == csi_cdla` |
| Create | `scripts/extract_representations.py` | Load checkpoint → extract [N, 256, 384] patch reps |
| Create | `scripts/segment_and_visualize.py` | Cluster + upsample + t-SNE + 5-row figure |

---

### Task 1: Generate CDL-A dataset

**Files:**
- Create: `data_gen/generate_csi_cdla.py`

**Verified Sionna output shapes** (from live API test):
```
a: [batch, 1, 1, 1, 32, 23, 1]   # 23 CDL-A paths, 32 TX antennas
tau: [batch, 1, 1, 23]             # path delays in seconds, max ≈ 966 ns
```

- [ ] **Step 1: Create `data_gen/` directory and write generation script**

```python
# data_gen/generate_csi_cdla.py
"""
Generate CDL-A CSI dataset for LatentMIM training.

Output files (in --out-dir):
  train_data.npy   [70000, 2, 64, 64] float32   real/imag delay-azimuth
  test_data.npy    [10000, 2, 64, 64] float32
  train_paths.npy  [70000]            int32      dominant path count per sample
  test_paths.npy   [10000]            int32
"""

import os
import argparse
import numpy as np

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
from sionna.phy.channel.tr38901 import AntennaArray, CDL

CARRIER_FREQUENCY = 7.0e9        # Hz
SUBCARRIER_SPACING = 30e3        # Hz  (unused for CIR path, kept for reference)
DELAY_SPREAD = 100e-9            # s   rms delay spread
SPEED = 30.0 / 3.6               # m/s  (30 km/h)
N_DELAY = 64                     # delay taps on output grid
N_FFT_ANGULAR = 64               # azimuth bins (2× zero-pad over 32 antennas)
MAX_DELAY = 1000e-9              # s   covers CDL-A max path delay (~966 ns)
DELAY_RESOLUTION = MAX_DELAY / N_DELAY  # ~15.625 ns per tap
PATH_THRESHOLD_DB = -20.0        # dB  threshold for dominant path count
N_TOTAL = 80_000
N_TEST = 10_000
N_TRAIN = N_TOTAL - N_TEST


def build_arrays():
    common = dict(
        polarization='single',
        polarization_type='H',
        antenna_pattern='38.901',
        carrier_frequency=CARRIER_FREQUENCY,
    )
    bs_array = AntennaArray(num_rows=1, num_cols=32, **common)
    ut_array = AntennaArray(num_rows=1, num_cols=1,  **common)
    return bs_array, ut_array


def process_batch(a_tf, tau_tf):
    """
    a_tf:   [B, 1, 1, 1, 32, 23, 1] complex
    tau_tf: [B, 1, 1, 23]            float  (seconds)

    Returns:
      data:    [B, 2, N_DELAY, N_FFT_ANGULAR] float32
      n_paths: [B]                             int32
    """
    a   = a_tf[:, 0, 0, 0, :, :, 0].numpy()  # [B, 32, 23] complex
    tau = tau_tf[:, 0, 0, :].numpy()          # [B, 23] seconds
    B   = a.shape[0]

    data    = np.zeros((B, 2, N_DELAY, N_FFT_ANGULAR), dtype=np.float32)
    n_paths = np.zeros(B, dtype=np.int32)

    for b in range(B):
        h   = a[b].T                           # [23, 32] complex
        tau_b = tau[b]                         # [23]

        # Count dominant paths
        power = np.mean(np.abs(h) ** 2, axis=-1)  # [23]
        max_p = power.max()
        if max_p > 0:
            threshold = max_p * 10 ** (PATH_THRESHOLD_DB / 10)
            n_paths[b] = int(np.sum(power >= threshold))

        # Angular domain: 1D FFT over 32 antennas → 64 azimuth bins
        H_angular = np.fft.fft(h, n=N_FFT_ANGULAR, axis=-1)  # [23, 64]

        # Delay grid: accumulate each path at nearest tap
        H_grid = np.zeros((N_DELAY, N_FFT_ANGULAR), dtype=complex)
        for l in range(23):
            tap = int(round(tau_b[l] / DELAY_RESOLUTION))
            if 0 <= tap < N_DELAY:
                H_grid[tap] += H_angular[l]

        data[b, 0] = H_grid.real
        data[b, 1] = H_grid.imag

    return data, n_paths


def generate(num_samples, channel_model, batch_size=64):
    all_data    = np.zeros((num_samples, 2, N_DELAY, N_FFT_ANGULAR), dtype=np.float32)
    all_n_paths = np.zeros(num_samples, dtype=np.int32)
    generated = 0

    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        a, tau = channel_model(batch_size=bs, num_time_steps=1,
                               sampling_frequency=1e9)
        data, n_paths = process_batch(a, tau)
        all_data[generated:generated + bs]    = data
        all_n_paths[generated:generated + bs] = n_paths
        generated += bs
        if generated % 5000 == 0 or generated == num_samples:
            print(f'  {generated}/{num_samples} samples generated')

    return all_data, all_n_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir',    default='data_gen/csi_cdla')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Building antenna arrays and CDL-A channel model...')
    bs_array, ut_array = build_arrays()
    channel_model = CDL(
        'A', DELAY_SPREAD, CARRIER_FREQUENCY,
        ut_array, bs_array,
        direction='downlink',
        min_speed=SPEED, max_speed=SPEED,
    )

    print(f'Generating {N_TOTAL} samples (batch_size={args.batch_size})...')
    all_data, all_n_paths = generate(N_TOTAL, channel_model, args.batch_size)

    # Shuffle before split
    rng = np.random.default_rng(42)
    idx = rng.permutation(N_TOTAL)
    all_data    = all_data[idx]
    all_n_paths = all_n_paths[idx]

    train_data, test_data       = all_data[:N_TRAIN],    all_data[N_TRAIN:]
    train_paths, test_paths     = all_n_paths[:N_TRAIN], all_n_paths[N_TRAIN:]

    np.save(os.path.join(args.out_dir, 'train_data.npy'),  train_data)
    np.save(os.path.join(args.out_dir, 'test_data.npy'),   test_data)
    np.save(os.path.join(args.out_dir, 'train_paths.npy'), train_paths)
    np.save(os.path.join(args.out_dir, 'test_paths.npy'),  test_paths)

    print(f'train_data:  {train_data.shape}  n_paths mean={train_paths.mean():.1f}')
    print(f'test_data:   {test_data.shape}   n_paths mean={test_paths.mean():.1f}')
    print(f'Saved to {args.out_dir}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run generation (expect ~20–30 min on V100)**

```bash
PYTHONPATH=. python data_gen/generate_csi_cdla.py --out-dir data_gen/csi_cdla --batch-size 64
```

Expected terminal output:
```
Building antenna arrays and CDL-A channel model...
Generating 80000 samples (batch_size=64)...
  5000/80000 samples generated
  ...
  80000/80000 samples generated
train_data:  (70000, 2, 64, 64)  n_paths mean=XX.X
test_data:   (10000, 2, 64, 64)  n_paths mean=XX.X
Saved to data_gen/csi_cdla
```

- [ ] **Step 3: Commit**

```bash
git add data_gen/generate_csi_cdla.py
git commit -m "feat: add CDL-A CSI dataset generation script"
```

---

### Task 2: Verify dataset

**Files:**
- Create: `data_gen/verify_csi_cdla.py`

- [ ] **Step 1: Write verification script**

```python
# data_gen/verify_csi_cdla.py
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
```

- [ ] **Step 2: Run verification**

```bash
PYTHONPATH=. python data_gen/verify_csi_cdla.py --data-dir data_gen/csi_cdla
```

Expected output:
```
Shapes OK
train n_paths: min=X max=X mean=XX.X
Saved visualisation to data_gen/csi_cdla/verify.png
```

Inspect `verify.png` — each heatmap should show sparse bright spots (path peaks) at various delay-azimuth positions.

- [ ] **Step 3: Commit**

```bash
git add data_gen/verify_csi_cdla.py
git commit -m "feat: add dataset verification script"
```

---

### Task 3: CSI dataset loader

**Files:**
- Modify: `datasets.py`

- [ ] **Step 1: Create `tests/` directory and write failing test**

```bash
mkdir -p tests && touch tests/__init__.py
```

```python
# tests/test_datasets_csi.py
import numpy as np
import os
import tempfile
import torch
import pytest

def make_fake_dataset(tmp_dir, n_train=10, n_test=4):
    rng = np.random.default_rng(0)
    np.save(os.path.join(tmp_dir, 'train_data.npy'),
            rng.standard_normal((n_train, 2, 64, 64)).astype(np.float32))
    np.save(os.path.join(tmp_dir, 'test_data.npy'),
            rng.standard_normal((n_test, 2, 64, 64)).astype(np.float32))
    np.save(os.path.join(tmp_dir, 'train_paths.npy'),
            rng.integers(2, 6, n_train).astype(np.int32))
    np.save(os.path.join(tmp_dir, 'test_paths.npy'),
            rng.integers(2, 6, n_test).astype(np.int32))


def test_csi_cdla_train_shape():
    import sys; sys.path.insert(0, '.')
    from datasets import csi_cdla
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(tmp)
        ds = csi_cdla(tmp, transform=None, train=True)
        assert len(ds) == 10
        x, label = ds[0]
        assert x.shape == (2, 64, 64)
        assert x.dtype == torch.float32


def test_csi_cdla_normalisation():
    import sys; sys.path.insert(0, '.')
    from datasets import csi_cdla
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(tmp)
        ds = csi_cdla(tmp, transform=None, train=True)
        x, _ = ds[0]
        power = torch.mean(x[0]**2 + x[1]**2).item()
        assert abs(power - 1.0) < 0.01  # normalised to unit power


def test_csi_cdla_test_split():
    import sys; sys.path.insert(0, '.')
    from datasets import csi_cdla
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(tmp)
        ds = csi_cdla(tmp, transform=None, train=False)
        assert len(ds) == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_datasets_csi.py -v
```

Expected: `ImportError` or `AttributeError: module 'datasets' has no attribute 'csi_cdla'`

- [ ] **Step 3: Add `CSICDLADataset` and `csi_cdla` to `datasets.py`**

Add after the `imagenet100` function (before `load_dataset`):

```python
class CSICDLADataset(data.Dataset):
    def __init__(self, data_path, train=True):
        split = 'train' if train else 'test'
        self.data    = np.load(os.path.join(data_path, f'{split}_data.npy'))   # [N, 2, 64, 64]
        self.n_paths = np.load(os.path.join(data_path, f'{split}_paths.npy'))  # [N]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)   # [2, 64, 64]
        p = np.sqrt(np.mean(x[0] ** 2 + x[1] ** 2))
        x = x / (p + 1e-8)
        return torch.from_numpy(x), int(self.n_paths[idx])


def csi_cdla(data_path, transform=None, train=True):
    return CSICDLADataset(data_path, train=train)
```

At the top of `datasets.py`, add the missing imports:
```python
import numpy as np
import torch
```
And add `'csi_cdla'` to the `__all__` list.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_datasets_csi.py -v
```

Expected:
```
PASSED tests/test_datasets_csi.py::test_csi_cdla_train_shape
PASSED tests/test_datasets_csi.py::test_csi_cdla_normalisation
PASSED tests/test_datasets_csi.py::test_csi_cdla_test_split
```

- [ ] **Step 5: Commit**

```bash
git add datasets.py tests/test_datasets_csi.py
git commit -m "feat: add CSICDLADataset loader with per-sample complex normalisation"
```

---

### Task 4: LatentMIM 2-channel patch embedding

**Files:**
- Modify: `models_lmim.py:27-33` (`LMIM.__init__`) and `models_lmim.py:186-192` (`build_lmim`)

- [ ] **Step 1: Write failing test**

```python
# tests/test_models_inchan2.py
import torch
import pytest

def test_lmim_2channel_forward():
    import sys; sys.path.insert(0, '.')
    from models_lmim import build_lmim
    model = build_lmim(
        'vit_small',
        in_chans=2,
        grid_size=16,
        loss='infonce_patches',
        tau=0.2,
        target_depth=12,
        decoder_depth=3,
        num_vis=25,
        avg_sim_coeff=0.0,
        avg_vis_mask_token=True,
        drop=0., attn_drop=0., drop_path=0.,
        freeze_pe=True,
        proj_cfg=type('P', (), {'mlp_dim': 4096, 'mlp_depth': 3})(),
        mask_target=True,
    )
    x = torch.randn(2, 2, 64, 64)
    loss, metrics = model(x, mom=0.99, sim_trg=0.75, update_ema=True)
    assert loss.item() > 0
    assert 'avg_sim_pred' in metrics
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models_inchan2.py -v
```

Expected: `TypeError: LMIM.__init__() got an unexpected keyword argument 'in_chans'`

- [ ] **Step 3: Add `in_chans` to `LMIM.__init__` in `models_lmim.py`**

Change the `__init__` signature (line 27) from:
```python
def __init__(self, grid_size=14, patch_size=16, patch_gap=0, in_chans=3,
```
to:
```python
def __init__(self, grid_size=14, patch_size=16, patch_gap=0, in_chans=2,
```

The two `PatchEmbed` calls (lines ~48 and ~56) already pass `in_chans` via:
```python
embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
```

Verify both encoder and target encoder lines pass `in_chans=in_chans`. If they currently hardcode `in_chans=3`, change each to `in_chans=in_chans`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_models_inchan2.py -v
```

Expected: `PASSED tests/test_models_inchan2.py::test_lmim_2channel_forward`

- [ ] **Step 5: Commit**

```bash
git add models_lmim.py tests/test_models_inchan2.py
git commit -m "feat: add in_chans parameter to LMIM for 2-channel CSI input"
```

---

### Task 5: CSI training config

**Files:**
- Create: `configs/lmim_csi.yaml`

- [ ] **Step 1: Write config file**

```yaml
# configs/lmim_csi.yaml
defaults:
  - hydra: default
  - env: default
  - log: default

worker: main_lmim
output_dir: ./checkpoints
job_name: latentMIM_csi
dataset: csi_cdla
data_path: data_gen/csi_cdla

encoder: vit_small
patch_size: 4
decoder_depth: 3
target_depth: 12
target_mom: 0.99
target_update_freq: 1
loss: infonce_patches
tau: 0.2

drop: 0.
attn_drop: 0.
drop_path: 0.

input_size: 64
min_crop: 1.0       # unused for CSI but required by config
grid_size: 16
num_vis: 25
patch_gap: 0
avg_vis_mask_token: True
avg_sim_coeff: 0.1
sim_init: 0.75
sim_end: 0.25
mask_target: True
freeze_pe: True

proj:
  mlp_dim: 4096
  mlp_depth: 3

epochs: 800
warmup_epochs: 40
start_epoch: 0
batch_size: 128
weight_decay: 0.05
blr: 1.5e-4
accum_iter: 1
lr:
min_lr_frac: 0.25
resume: True

knn_eval_only: False
eval_freq: 10

log:
  print_freq: 547    # ≈ steps per epoch (70000 / 128)
  save_freq: 50
  use_wandb: False
```

- [ ] **Step 2: Commit**

```bash
git add configs/lmim_csi.yaml
git commit -m "feat: add CSI LatentMIM training config"
```

---

### Task 6: Replace ImageNet transforms in training loop

**Files:**
- Modify: `main_lmim.py:65-86`

- [ ] **Step 1: Write failing test**

```python
# tests/test_main_transforms.py
import sys; sys.path.insert(0, '.')

def test_csi_transform_is_none(monkeypatch):
    """When dataset=csi_cdla, transforms passed to load_dataset should be None."""
    import types

    captured = {}

    import datasets as ds_module
    original_load = ds_module.load_dataset
    def mock_load(dataset, path, transform, train=True):
        captured['transform'] = transform
        captured['dataset']   = dataset
        # return a trivial fake dataset
        class FakeDS:
            def __len__(self): return 10
        return FakeDS()
    monkeypatch.setattr(ds_module, 'load_dataset', mock_load)

    import argparse
    args = argparse.Namespace(
        dataset='csi_cdla', data_path='/tmp',
        grid_size=16, patch_size=4, patch_gap=0,
        min_crop=1.0,
    )

    # Import the helper function we will extract from main_lmim.py
    from main_lmim import build_transforms
    train_t, eval_t = build_transforms(args)
    assert train_t is None
    assert eval_t is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_main_transforms.py -v
```

Expected: `ImportError: cannot import name 'build_transforms' from 'main_lmim'`

- [ ] **Step 3: Extract `build_transforms` and update `main_lmim.py`**

Add this function directly before `main_worker` in `main_lmim.py`:

```python
def build_transforms(args):
    """Return (train_transform, eval_transform). Both are None for CSI datasets."""
    if args.dataset == 'csi_cdla':
        return None, None

    train_img_size = args.grid_size * (args.patch_size + args.patch_gap)
    eval_img_size  = args.grid_size * args.patch_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(train_img_size, scale=(args.min_crop, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    eval_transform = transforms.Compose([
        transforms.Resize(int(eval_img_size / 0.875),
                          interpolation=F.InterpolationMode.BICUBIC),
        transforms.CenterCrop(eval_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_transform, eval_transform
```

Then in `main_worker`, replace the existing transform + dataset construction block (lines 65-86) with:

```python
train_transform, eval_transform = build_transforms(args)
db_train = datasets.load_dataset(
    args.dataset, args.data_path, transform=train_transform, train=True)
db_eval  = datasets.load_dataset(
    args.dataset, args.data_path, transform=eval_transform,  train=False)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_main_transforms.py -v
```

Expected: `PASSED tests/test_main_transforms.py::test_csi_transform_is_none`

- [ ] **Step 5: Commit**

```bash
git add main_lmim.py tests/test_main_transforms.py
git commit -m "feat: skip ImageNet transforms for CSI datasets"
```

---

### Task 7: Train LatentMIM on CSI data

**Files:**
- No new files. Uses `launcher.py` + `configs/lmim_csi.yaml`.

- [ ] **Step 1: Launch training (single GPU, no Slurm)**

```bash
PYTHONPATH=. python launcher.py --config-name=lmim_csi \
  env.slurm=False env.distributed=False env.ngpu=1 \
  data_path=data_gen/csi_cdla \
  output_dir=./checkpoints
```

Expected: training starts, one loss summary line per epoch, KNN accuracy printed every 10 epochs:
```
Epoch: [0]  loss: X.XXX  lr: X.XXXXXX
...
Epoch: [10] NN Acc: XX.X%
```

- [ ] **Step 2: Monitor convergence**

Watch KNN accuracy in terminal every 10 epochs. Stop manually when it plateaus for 20–30 consecutive evaluations. Checkpoint is saved to `./checkpoints/latentMIM_csi/checkpoints/`.

---

### Task 8: Extract patch-level representations

**Files:**
- Create: `scripts/extract_representations.py`

- [ ] **Step 1: Write extraction script**

```python
# scripts/extract_representations.py
"""
Extract patch-level representations from the online encoder for the test set.

Usage:
  python scripts/extract_representations.py \
    --ckpt checkpoints/latentMIM_csi/checkpoints/checkpoint_XXXX.pth \
    --data-dir data_gen/csi_cdla \
    --out scripts/representations.npy
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '.')

from datasets import csi_cdla
from models_lmim import build_lmim


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Build ViT-Small with CSI config
    import types
    proj_cfg = types.SimpleNamespace(mlp_dim=4096, mlp_depth=3)
    model = build_lmim(
        'vit_small',
        in_chans=2,
        grid_size=16,
        loss='infonce_patches',
        tau=0.2,
        target_depth=12,
        decoder_depth=3,
        num_vis=25,
        avg_sim_coeff=0.0,
        avg_vis_mask_token=True,
        drop=0., attn_drop=0., drop_path=0.,
        freeze_pe=True,
        proj_cfg=proj_cfg,
        mask_target=True,
    )
    state = ckpt['state_dict']
    # Strip DDP prefix if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def extract(model, loader, device):
    all_reps = []
    for imgs, _ in loader:
        imgs = imgs.to(device)                        # [B, 2, 64, 64]
        # VIT.forward handles 4D input: patchifies internally, computes pid
        reps = model.encoder(imgs)                    # [B, 257, 384] (CLS + 256 patches)
        reps = reps[:, 1:, :]                         # drop CLS → [B, 256, 384]
        all_reps.append(reps.cpu().numpy())
    return np.concatenate(all_reps, axis=0)           # [N, 256, 384]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     required=True)
    parser.add_argument('--data-dir', default='data_gen/csi_cdla')
    parser.add_argument('--out',      default='scripts/representations.npy')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)

    test_ds = csi_cdla(args.data_dir, train=False)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f'Extracting representations for {len(test_ds)} test samples...')
    reps = extract(model, loader, device)
    assert reps.shape == (len(test_ds), 256, 384), f'unexpected shape: {reps.shape}'
    print(f'Representations shape: {reps.shape}')   # (10000, 256, 384)

    np.save(args.out, reps)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run extraction**

```bash
python scripts/extract_representations.py \
  --ckpt checkpoints/latentMIM_csi/checkpoints/checkpoint_XXXX.pth \
  --data-dir data_gen/csi_cdla \
  --out scripts/representations.npy
```

Expected:
```
Extracting representations for 10000 test samples...
Representations shape: (10000, 256, 384)
Saved to scripts/representations.npy
```

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_representations.py
git commit -m "feat: add patch representation extraction script"
```

---

### Task 9: Segmentation and visualisation

**Files:**
- Create: `scripts/segment_and_visualize.py`

- [ ] **Step 1: Write segmentation and visualisation script**

```python
# scripts/segment_and_visualize.py
"""
For 5 random test samples:
  1. Run AgglomerativeClustering on [256, 384] patch representations
  2. Upsample 16×16 label map → 64×64 (nearest neighbour)
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
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.ndimage import zoom

N_SAMPLES  = 5
PATCH_GRID = 16   # 16×16 = 256 patches
K_MIN, K_MAX = 2, 5


def cluster_sample(reps_256_384, n_paths):
    """
    reps_256_384: [256, 384]
    n_paths: int  (dominant path count, clamped to [K_MIN, K_MAX])
    Returns: labels [256] int, k int
    """
    k = int(np.clip(n_paths, K_MIN, K_MAX))
    labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(reps_256_384)
    return labels, k


def upsample_labels(labels_256, patch_grid=PATCH_GRID, target=64):
    """
    labels_256: [256] int  →  label_map [target, target] int
    """
    label_map = labels_256.reshape(patch_grid, patch_grid)
    scale = target / patch_grid
    # nearest-neighbour: zoom with order=0
    upsampled = zoom(label_map.astype(float), scale, order=0).astype(int)
    return upsampled


def run_tsne(reps_256_384):
    """Returns [256, 2] 2-D embedding."""
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
        rep   = reps[idx]           # [256, 384]
        x     = test_data[idx]      # [2, 64, 64]
        k_val = int(n_paths[idx])

        labels, k = cluster_sample(rep, k_val)
        label_map = upsample_labels(labels)   # [64, 64]
        tsne_xy   = run_tsne(rep)             # [256, 2]
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
                              for c in row_labels]
                             for row_labels in label_map])
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
```

- [ ] **Step 2: Run visualisation**

```bash
python scripts/segment_and_visualize.py \
  --reps  scripts/representations.npy \
  --data  data_gen/csi_cdla/test_data.npy \
  --paths data_gen/csi_cdla/test_paths.npy \
  --out   scripts/segmentation_results.png
```

Expected: `Saved to scripts/segmentation_results.png`

Inspect the figure: cluster colours in the segmentation map and t-SNE should correspond to the same propagation paths visible in the CSI magnitude heatmap.

- [ ] **Step 3: Commit**

```bash
git add scripts/segment_and_visualize.py
git commit -m "feat: add segmentation and visualisation script"
```
