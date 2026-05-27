import numpy as np
import os
import tempfile
import torch
import pytest


def make_fake_dataset(tmp_dir, n_train=10, n_test=4):
    rng = np.random.default_rng(0)
    # dB magnitude: shape [N, 1, 64, 64], values roughly in range [-80, 0] dB
    np.save(os.path.join(tmp_dir, 'train_data.npy'),
            rng.standard_normal((n_train, 1, 64, 64)).astype(np.float32) * 20 - 40)
    np.save(os.path.join(tmp_dir, 'test_data.npy'),
            rng.standard_normal((n_test, 1, 64, 64)).astype(np.float32) * 20 - 40)
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
        assert x.shape == (1, 64, 64)
        assert x.dtype == torch.float32


def test_csi_cdla_normalisation():
    import sys; sys.path.insert(0, '.')
    from datasets import csi_cdla
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(tmp)
        ds = csi_cdla(tmp, transform=None, train=True)
        x, _ = ds[0]
        assert abs(x.mean().item()) < 0.1       # approximately zero mean
        assert abs(x.std().item() - 1.0) < 0.1  # approximately unit std


def test_csi_cdla_test_split():
    import sys; sys.path.insert(0, '.')
    from datasets import csi_cdla
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(tmp)
        ds = csi_cdla(tmp, transform=None, train=False)
        assert len(ds) == 4
