"""Dynamic Spatial Filtering on synthetic tensors, alone and in front of the CNN and SNN."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dsf import (DynamicSpatialFilter, ImportanceProbe, channel_importance, dsf_spec,
                            soft_threshold, spatial_features)
from src.models.registry import build_model
from src.models.snn import load_snn_config

C, T = 18, 1024
CNN_CFG = dict(n_temporal_filters=2, temporal_kernel_size=5, pool_size=10, pool_stride=5, dropout=0.)


def masked_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, C, T, generator=g)
    x[1, :5] = 0    # 25%
    x[2, :14] = 0   # 75%
    x[3] = 0        # everything missing
    return x


@pytest.mark.parametrize("variant", ["dsfd", "dsfm_st"])
def test_preserves_shape_and_is_finite_under_channel_loss(variant):
    torch.manual_seed(0)
    dsf = DynamicSpatialFilter.from_spec(dsf_spec(variant), C)
    out = dsf(masked_batch())
    assert out.shape == (4, C, T) and out.dtype == torch.float32 and torch.isfinite(out).all()


def test_identity_filters_reproduce_input():
    """Reference slicing: the first C' MLP outputs are the bias, the rest W row-major [C', C]."""
    dsf = DynamicSpatialFilter(C, features="logvar")
    last = dsf.mlp[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.cat([torch.arange(C, dtype=torch.float32), torch.eye(C).flatten()]))
    x = torch.randn(2, C, T)
    torch.testing.assert_close(dsf(x), x + torch.arange(C, dtype=torch.float32)[:, None])


def test_soft_threshold_matches_paper_equation():
    w = torch.tensor([-0.5, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
    expected = torch.sign(w) * torch.clamp(w.abs() - 0.1, min=0)
    torch.testing.assert_close(soft_threshold(w, 0.1), expected)
    torch.testing.assert_close(soft_threshold(w, 0.1), torch.tensor([-0.4, 0, 0, 0, 0, 0, 0.2]))


def test_logm_features_match_numpy():
    x = torch.randn(3, C, T, dtype=torch.float64)
    feats = spatial_features(x, "logm")
    rows, cols = np.triu_indices(C)
    for i in range(3):
        e, v = np.linalg.eigh(np.cov(x[i].numpy()))
        expected = (v @ np.diag(np.log(e)) @ v.T)[rows, cols]
        np.testing.assert_allclose(feats[i].numpy(), expected, rtol=1e-6, atol=1e-8)


def test_logvar_features_and_flat_channel_handling():
    x = masked_batch().double()
    clamp = spatial_features(x, "logvar", "clamp")
    torch.testing.assert_close(clamp[0], x[0].var(-1).log())
    # Masked (zeroed) channels are floored at log(1e-10) and stand apart from real channels.
    assert torch.allclose(clamp[2, :14], torch.full((14,), np.log(1e-10), dtype=torch.float64))
    assert (clamp[2, 14:] > -5).all()
    # Paper/reference rule: log(0) -> 0, indistinguishable from a unit-variance channel.
    zero = spatial_features(x, "logvar", "zero")
    assert (zero[2, :14] == 0).all()


def test_logm_marks_masked_channels():
    x = masked_batch()
    feats = spatial_features(x, "logm")
    rows, cols = torch.triu_indices(C, C)
    diagonal = rows == cols
    masked_diag = diagonal & (rows < 14)
    assert torch.allclose(feats[2, masked_diag], torch.tensor(np.log(1e-10), dtype=torch.float32), atol=1e-3)
    # Off-diagonal entries involving a masked channel are zero.
    off = ~diagonal & ((rows < 14) | (cols < 14))
    assert feats[2, off].abs().max() < 1e-4


@pytest.mark.parametrize("variant", ["dsfd", "dsfm_st"])
def test_gradients_reach_mlp_with_degenerate_inputs(variant):
    """Repeated zero eigenvalues (many masked channels) must not produce NaN gradients."""
    torch.manual_seed(0)
    dsf = DynamicSpatialFilter.from_spec(dsf_spec(variant), C)
    dsf(masked_batch()).pow(2).mean().backward()
    for name, p in dsf.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_spec_defaults_follow_paper():
    assert dsf_spec("none") is None and dsf_spec(None) is None
    d, m = dsf_spec("dsfd"), dsf_spec("dsfm_st")
    assert (d["features"], d["soft_threshold"], d["hidden"]) == ("logvar", None, C * C)
    assert (m["features"], m["soft_threshold"]) == ("logm", 0.1)
    # Hidden layer of C^2 units: C=4 log-variance gives the paper's smallest module (420 params).
    small = DynamicSpatialFilter.from_spec(dsf_spec("dsfd", 4), 4)
    assert sum(p.numel() for p in small.parameters()) == 420
    with pytest.raises(ValueError):
        dsf_spec("dsfx")


def test_channel_importance_is_column_norm():
    w = torch.zeros(1, 3, 3)
    w[0, :, 0] = torch.tensor([3.0, 4.0, 0.0])  # ||col 0|| = 5
    w[0, 1, 2] = 1.0
    torch.testing.assert_close(channel_importance(w, normalize=False), torch.tensor([[5.0, 0.0, 1.0]]))
    torch.testing.assert_close(channel_importance(w), torch.tensor([[1.0, 0.0, 0.2]]))
    assert torch.isfinite(channel_importance(torch.zeros(1, 3, 3))).all()


def test_importance_probe_splits_available_and_missing():
    torch.manual_seed(0)
    dsf = DynamicSpatialFilter.from_spec(dsf_spec("dsfd"), C)
    probe = ImportanceProbe(dsf)
    x = masked_batch()[:3]
    with torch.no_grad():
        dsf(x)
        phi = channel_importance(dsf.filters(x)[0])
    s = probe.summary()
    missing = x.abs().amax(-1) == 0
    assert s["missing_mean"] == pytest.approx(float(phi[missing].double().mean()), rel=1e-6)
    assert s["available_mean"] == pytest.approx(float(phi[~missing].double().mean()), rel=1e-6)
    assert s["missing"][17] is None  # channel 17 was never masked
    probe.reset()
    assert probe.summary()["missing_mean"] is None
    probe.remove()


@pytest.mark.parametrize("model_type", ["cnn", "snn"])
def test_registry_attaches_dsf_and_round_trips(model_type, tmp_path):
    model_cfg = CNN_CFG if model_type == "cnn" else load_snn_config()["model"]
    plain_cfg = dict(model_type=model_type, model=model_cfg)
    dsf_cfg = dict(plain_cfg, dsf=dsf_spec("dsfm_st"))
    torch.manual_seed(0)
    plain = build_model(plain_cfg)
    torch.manual_seed(0)
    model = build_model(dsf_cfg)
    assert isinstance(model.front_end, DynamicSpatialFilter)
    # The DSF is created after the backbone: same seed -> same backbone initialization.
    for (name, a), b in zip(plain.backbone.state_dict().items(), model.backbone.state_dict().values()):
        torch.testing.assert_close(a, b, msg=name)

    x = masked_batch()
    model.train()
    torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.front_end.parameters())

    path = tmp_path / "m.pt"
    torch.save(dict(config=dsf_cfg, model_state_dict=model.state_dict()), path)
    payload = torch.load(path, weights_only=True)
    restored = build_model(payload["config"])
    restored.load_state_dict(payload["model_state_dict"])
    model.eval(), restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))
    with pytest.raises(ValueError, match="DSF"):
        build_model(dsf_cfg, front_end=torch.nn.Identity())


@pytest.mark.parametrize("model_type", ["cnn", "snn"])
def test_pipeline_evaluates_dsf_checkpoint(model_type, tmp_path):
    from src.evaluation.pipeline import run
    from tests.test_evaluation_pipeline import make_experiment
    arguments = make_experiment(tmp_path)
    saved = torch.load(arguments.checkpoint, weights_only=True)
    model_cfg = saved["config"]["model"] if model_type == "cnn" else load_snn_config()["model"]
    config = dict(saved["config"], model_type=model_type, model=model_cfg, dsf=dsf_spec("dsfd"))
    torch.manual_seed(9)
    saved.update(model_state_dict=build_model(config).state_dict(), config=config)
    torch.save(saved, arguments.checkpoint)
    run(arguments)
    arguments.stage, arguments.calibration = "test", arguments.output / "calibration.json"
    arguments.output = tmp_path / "test_results"
    run(arguments)
    pooled = pd.read_csv(arguments.output / "pooled_metrics.csv")
    assert pooled.dsf_importance_available.between(0, 1).all()
    assert pooled.dsf_importance_missing.isna().iloc[0]  # complete-channel condition
    assert pooled.dsf_importance_missing.iloc[1:].between(0, 1).all()
    importance = pd.read_csv(arguments.output / "dsf_channel_importance.csv")
    assert len(importance) == 7 * C
    summary = json.loads((arguments.output / "summary.json").read_text())
    assert summary["dsf"]["variant"] == "dsfd"
    timing = summary["timing"]
    assert timing["front_end_parameters"] > 0
    assert timing["front_end_parameters"] + timing["backbone_parameters"] == timing["parameters"]
    assert summary["energy"]["front_end_macs"] > 0
