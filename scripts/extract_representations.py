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
import types
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '.')

from datasets import csi_cdla
from models_lmim import build_lmim


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    proj_cfg = types.SimpleNamespace(mlp_dim=4096, mlp_depth=3)
    model = build_lmim(
        'vit_small',
        patch_size=4,
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
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def extract(model, loader, device):
    all_reps = []
    for imgs, _ in loader:
        imgs = imgs.to(device)                        # [B, 2, 64, 64]
        reps = model.encoder(imgs)                    # [B, 257, 384] (CLS + 256 patches)
        reps = reps[:, 1:, :]                         # drop CLS → [B, 256, 384]
        all_reps.append(reps.cpu().numpy())
    return np.concatenate(all_reps, axis=0)           # [N, 256, 384]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--data-dir',   default='data_gen/csi_cdla')
    parser.add_argument('--out',        default='scripts/representations.npy')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)

    test_ds = csi_cdla(args.data_dir, train=False)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f'Extracting representations for {len(test_ds)} test samples...')
    reps = extract(model, loader, device)
    assert reps.shape == (len(test_ds), 256, 384), f'unexpected shape: {reps.shape}'
    print(f'Representations shape: {reps.shape}')

    np.save(args.out, reps)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
