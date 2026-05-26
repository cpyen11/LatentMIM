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
