"""Compact EESNN on synthetic tensors; no CHB-MIT data needed."""
import math

import pytest
import torch

from src.models.snn import CompactEESNN, load_snn_config

C, T = 18, 1024


@pytest.fixture(scope="module")
def cfg():
    return load_snn_config()["model"]


def make(cfg, **overrides):
    torch.manual_seed(0)
    return CompactEESNN({**cfg, **overrides})


@pytest.mark.parametrize("n_hidden", [1, 2])
def test_output_shape_and_rates(cfg, n_hidden):
    model = make(cfg, n_hidden=n_hidden)
    logits, rates = model(torch.randn(4, C, T), return_rates=True)
    assert logits.shape == (4,) and torch.isfinite(logits).all()
    assert len(rates) == n_hidden
    for r in rates:
        assert r.shape == (4,) and ((r >= 0) & (r <= 1)).all()


@pytest.mark.parametrize("n_hidden", [1, 2])
def test_zeroed_channels_stay_finite(cfg, n_hidden):
    """Missing channels are zeroed by the masking simulator; an all-zero window included."""
    model = make(cfg, n_hidden=n_hidden)
    x = torch.randn(3, C, T)
    x[0, :14] = 0.0  # ~75% missing
    x[1] = 0.0       # everything missing
    assert torch.isfinite(model(x)).all()
    model.eval()
    assert torch.isfinite(model(x)).all()


@pytest.mark.parametrize(
    "input_kernel, input_stride, hidden_stride",
    [([5, 20], [1, 4], [1, 2]), ([5, 5], [1, 1], [1, 2]), ([3, 7], [1, 3], [1, 2]), ([5, 20], [1, 4], [2, 2])],
)
@pytest.mark.parametrize("n_hidden", [1, 2])
def test_resolution_is_config_only(cfg, n_hidden, input_kernel, input_stride, hidden_stride):
    """Other kernels/strides need no code change: feedback and readout adapt to the shapes."""
    model = make(cfg, n_hidden=n_hidden, input_kernel=input_kernel,
                 input_stride=input_stride, hidden_stride=hidden_stride)
    assert model(torch.randn(2, C, T)).shape == (2,)


def test_layer_shapes_default(cfg):
    model = make(cfg, n_hidden=2)
    assert model.layer_shapes == [(18, 256), (18, 128)]


@pytest.mark.parametrize("n_hidden", [1, 2])
def test_gradients_reach_every_layer(cfg, n_hidden):
    model = make(cfg, n_hidden=n_hidden)
    model(torch.randn(8, C, T) * 3).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("n_hidden", [1, 2])
def test_feedback_used_only_after_first_step(cfg, n_hidden):
    # Train mode: BN uses batch statistics, so layers spike as they do during training
    # (untrained running statistics in eval mode can leave the last layer silent).
    x = torch.randn(4, C, T)
    model = make(cfg, n_hidden=n_hidden, feedback_spectral_norm=False)
    with torch.no_grad():
        y, rates = model(x, return_rates=True)
        assert rates[-1].sum() > 0, "last layer must spike for the feedback to matter"
        model.feedback.weight.zero_()
        y_no_fb = model(x)
        assert not torch.allclose(y, y_no_fb), "feedback should change the output when T=2"

        single = make(cfg, n_hidden=n_hidden, feedback_spectral_norm=False, time_steps=1)
        y1 = single(x)
        single.feedback.weight.normal_()
        torch.testing.assert_close(single(x), y1)  # T=1 never reaches the feedback


def test_readout_is_normalized_rate(cfg):
    """With every neuron firing at every step, the pooled rate fed to the readout is 1."""
    model = make(cfg, n_hidden=1)
    seen = {}
    model.readout.register_forward_hook(lambda m, inp, out: seen.update(x=inp[0]))
    model.v_th = -1e9  # always fire
    model(torch.randn(2, C, T))
    torch.testing.assert_close(seen["x"], torch.ones_like(seen["x"]))


def test_eval_is_deterministic(cfg):
    model = make(cfg, n_hidden=2)
    model.eval()
    x = torch.randn(4, C, T)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x))


def test_param_count_is_compact(cfg):
    n1 = make(cfg, n_hidden=1).count_params()
    n2 = make(cfg, n_hidden=2).count_params()
    assert n1 < n2 < 100_000


def test_rejects_unsupported_settings(cfg):
    with pytest.raises(ValueError):
        CompactEESNN({**cfg, "dropout": 0.2})
    with pytest.raises(ValueError):
        CompactEESNN({**cfg, "surrogate": "atan"})
    with pytest.raises(ValueError):
        CompactEESNN({**cfg, "n_hidden": 3})


@pytest.mark.parametrize("n_hidden", [1, 2])
def test_overfits_small_synthetic_task(cfg, n_hidden):
    """32 noise windows; positives carry a 3 Hz rhythm on 6 channels. The model must fit them
    almost perfectly -- if it cannot, training is broken, independent of real data."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, C, T, generator=g)
    y = torch.arange(32) % 2
    t = torch.arange(T) / 256.0
    x[y == 1, :6] += 2.0 * torch.sin(2 * math.pi * 3.0 * t)

    model = make(cfg, n_hidden=n_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(x), y.float())
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    model.eval()
    with torch.no_grad():
        assert ((model(x) > 0).long() == y).all()
