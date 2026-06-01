# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latent Masked Image Modeling (Latent MIM) is a self-supervised visual representation learning method (ECCV 2024) that reconstructs masked image patch features in latent space using a ViT backbone. It uses a momentum-updated target encoder (EMA) to avoid representation collapse, and an InfoNCE patch-level loss to prevent trivial solutions.

## Running Training

**Local multi-GPU (no Slurm):**
```bash
PYTHONPATH=. python launcher.py --config-name=lmim \
  worker=main_lmim \
  encoder=vit_base decoder_depth=3 avg_sim_coeff=0.1 loss=infonce_patches patch_gap=4 \
  epochs=800 warmup_epochs=40 blr=1.5e-4 min_lr_frac=0.25 weight_decay=0.05 \
  batch_size=256 accum_iter=2 env.ngpu=8 \
  dataset=imagenet resume=True \
  output_dir=./checkpoints \
  data_path=/path/to/imagenet \
  env.slurm=False env.distributed=True
```

**Slurm cluster:**
```bash
PYTHONPATH=. python launcher.py --config-name=lmim \
  ... env.slurm=True env.distributed=True
```

**KNN eval only (from checkpoint):**
```bash
PYTHONPATH=. python launcher.py --config-name=lmim \
  knn_eval_only=True resume=True output_dir=./checkpoints data_path=/path/to/imagenet \
  env.slurm=False env.distributed=False
```

## Architecture

The system has three main components, all defined in `models_lmim.py`:

**LMIM model** (`models_lmim.py`):
- `encoder` (VIT): Online encoder, processes only visible patches + CLS token
- `target_encoder` (VIT): EMA copy of encoder, updated with momentum `m` each step; processes target patches; gradients disabled
- `decoder` (CrossDecoder): Takes visible encoder output as key/value and learned mask tokens as queries, predicts latent features for masked patches

**Forward pass flow** (`LMIM.forward`):
1. `Patchify` splits image into patches (with optional random `patch_gap` offset for data augmentation)
2. Patches are randomly split into visible (`num_vis`) and masked sets
3. Target encoder (no grad) processes target patches → supervision signal
4. Online encoder processes visible patches
5. CrossDecoder takes encoder output + mask position embeddings → patch predictions
6. InfoNCE loss compares predictions against target encoder features

**ViT building blocks** (`vits.py`):
- `Patchify`: Converts images to patch tensors; `patch_gap > 0` randomly offsets the extraction window per patch, acting as a stochastic augmentation
- `AbsolutePositionEmbeds`: Frozen sincos 2D positional embeddings indexed by patch ID
- `CrossPositionAttentionStem`: Initializes mask tokens by attending over visible patch features weighted by positional similarity

**Cross-attention decoder** (`xvits.py`):
- `CrossDecoder`: Stack of `CrossBlock` layers; each block does self-attention on mask tokens then cross-attention to encoder output
- Mask tokens are initialized with position-weighted averages of visible features via `CrossPositionAttentionStem`

## Configuration

Config is managed by **Hydra** (`configs/lmim.yaml`). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder` | `vit_base` | Backbone: `vit_tiny/small/base/large/huge` |
| `decoder_depth` | 3 | Number of CrossDecoder layers |
| `target_depth` | 12 | Target encoder depth (0 = shallow target) |
| `patch_gap` | 4 | Gap between patches (0 = standard ViT) |
| `num_vis` | 20 | Number of visible patches (out of 196 for 14×14 grid) |
| `loss` | `infonce_patches` | Loss type: `infonce_patches` or `norm_l2` |
| `avg_sim_coeff` | 0 | Coefficient for similarity regularization loss |
| `tau` | 0.2 | InfoNCE temperature |
| `target_mom` | 0.99 | Starting EMA momentum (cosine-scheduled up to 1.0) |
| `freeze_pe` | True | Freeze positional embeddings (buffers, not parameters) |

Override any parameter on the command line: `launcher.py --config-name=lmim key=value`.

## Data

`datasets.py` supports `imagenet` and `imagenet100`. The `imagenet` loader expects:
- `data_path/train/` — ImageNet training images in synset folders
- `data_path/val/ILSVRC2012_val_XXXXXXXX.JPEG` — flat validation images
- `data_path/anno/meta.mat` and `data_path/anno/ILSVRC2012_validation_ground_truth.txt`

## Utilities

- `util/misc.py`: `CheckpointManager` (save/resume), `eval_knn` (KNN evaluation), distributed init, WandB init, `MetricLogger`
- `util/lr_sched.py`: Cosine LR schedule with linear warmup
- `util/dist_utils.py`: Distributed training helpers
- `launcher.py`: Submitit-based Slurm launcher; skips jobs already queued or whose final checkpoint exists

## Dependencies

Python 3.8+, PyTorch 2.1, torchvision, timm, hydra-core, numpy, scipy, submitit, wandb
