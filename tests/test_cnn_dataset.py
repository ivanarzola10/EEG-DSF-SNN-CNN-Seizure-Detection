import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preparation import ChannelMoments, validate_splits
from src.data.torch_dataset import WindowDataset
from src.models.cnn import ShallowEEGCNN


def test_identity_leakage():
    with pytest.raises(ValueError, match="identity"):
        validate_splits(dict(train=["chb01"], val=["chb02"], test=["chb21"]), {"chb21": "chb01"})


def test_continuous_statistics():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(18, 2700))
    moments = ChannelMoments()
    moments.update(x[:, :600])
    moments.update(x[:, 600:])
    mean, std = moments.finish()
    np.testing.assert_allclose(mean[:, 0], x.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(std[:, 0], x.std(axis=1), rtol=1e-6)


def fixture_dataset(tmp_path, legacy=False):
    x = np.arange(18 * 1280, dtype=np.float32).reshape(18, 1280)
    row = dict(patient_id="chb02", file_id="chb02_01", start_s=1., end_s=5., label=1)
    if legacy:
        np.savez_compressed(tmp_path / "data.npz", X=x[:, 256:][None], y=np.array([1]))
        row.update(npz_path="data.npz", window_index=0)
    else:
        np.save(tmp_path / "data.npy", x)
        row.update(continuous_path="data.npy", start_sample=256)
    pd.DataFrame([row, {**row, "patient_id": "chb03"}]).to_csv(tmp_path / "meta.csv", index=False)
    np.savez(tmp_path / "norm.npz", mean=np.ones((18, 1)), std=np.full((18, 1), 2.))
    return x


@pytest.mark.parametrize("legacy", [False, True])
def test_loader_normalization_metadata_and_isolation(tmp_path, legacy):
    x = fixture_dataset(tmp_path, legacy)
    ds = WindowDataset(tmp_path / "meta.csv", ["chb02"], tmp_path / "norm.npz", return_metadata=True)
    assert len(ds) == 1
    tensor, y, metadata = ds[0]
    np.testing.assert_array_equal(tensor.numpy(), (x[:, 256:] - 1) / 2)
    assert y.dtype == torch.long and y.item() == 1
    assert metadata["end_s"] == 5.
    tensor.zero_()
    assert ds[0][0].abs().sum() > 0  # Caller cannot mutate cached data.
    assert len(ds._cache) == 1
    assert ds.__getstate__()["_cache"] == {}


def test_normalization_must_be_explicit(tmp_path):
    fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match="explicitly"):
        WindowDataset(tmp_path / "meta.csv", ["chb02"])
    np.savez(tmp_path / "bad.npz", mean=np.zeros((18, 1)), std=np.zeros((18, 1)))
    with pytest.raises(ValueError, match="positive"):
        WindowDataset(tmp_path / "meta.csv", ["chb02"], tmp_path / "bad.npz")


def test_real_shallownet_backward_and_reload():
    from braindecode.models import ShallowFBCSPNet
    torch.manual_seed(7)
    model = ShallowEEGCNN()
    assert isinstance(model.backbone, ShallowFBCSPNet)
    x = torch.randn(2, 18, 1024)
    logits = model(x)
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    restored = ShallowEEGCNN().eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), restored(x))


def test_prepared_manifest_rejects_foreign_normalization(tmp_path):
    import json
    fixture_dataset(tmp_path)
    manifest = dict(status="complete", data_config={"montage": {"derivations": [f"c{i}" for i in range(18)]}},
                    splits={"train": ["chb02"]}, normalization_records=["chb02_01"])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "norm.npz", mean=np.zeros((18, 1)), std=np.ones((18, 1)),
             channels=np.array([f"c{i}" for i in range(18)]), train_patients=np.array(["chb03"]),
             records=np.array(["chb03_01"]))
    with pytest.raises(ValueError, match="training patients"):
        WindowDataset(tmp_path / "meta.csv", ["chb02"], tmp_path / "norm.npz")
    manifest["status"] = "failed"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not completed"):
        WindowDataset(tmp_path / "meta.csv", ["chb02"], normalize=False)
