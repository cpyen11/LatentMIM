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
    x = torch.randn(2, 2, 256, 256)
    loss, metrics = model(x, mom=0.99, sim_trg=0.75, update_ema=True)
    assert loss.item() > 0
    assert 'avg_sim_pred' in metrics
