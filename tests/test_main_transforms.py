import sys; sys.path.insert(0, '.')
import types, importlib


def test_csi_transform_is_none(monkeypatch):
    """When dataset=csi_cdla, transforms passed to load_dataset should be None."""
    import datasets as ds_module

    def mock_load(dataset, path, transform, train=True):
        class FakeDS:
            def __len__(self): return 10
        return FakeDS()
    monkeypatch.setattr(ds_module, 'load_dataset', mock_load)

    # Stub wandb so main_lmim can be imported without the package installed
    if 'wandb' not in sys.modules:
        sys.modules['wandb'] = types.ModuleType('wandb')

    import argparse
    args = argparse.Namespace(
        dataset='csi_cdla', data_path='/tmp',
        grid_size=16, patch_size=4, patch_gap=0,
        min_crop=1.0,
    )

    from main_lmim import build_transforms
    train_t, eval_t = build_transforms(args)
    assert train_t is None
    assert eval_t is None
