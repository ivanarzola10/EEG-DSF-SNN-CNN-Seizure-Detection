import pytest
import torch

from src.models.lif import lif_step, spike

V_TH = 2.0
LEAK = 0.95


def test_hand_computed_sequence():
    """Constant current 1.5 for three steps, checked against Eq. (1) by hand."""
    u = torch.zeros(1)
    current = torch.full((1,), 1.5)
    # t=1: u_half = 1.5            -> no spike, u = 1.5
    # t=2: u_half = 0.95*1.5 + 1.5 = 2.925   -> spike, u = 0.925
    # t=3: u_half = 0.95*0.925 + 1.5 = 2.37875 -> spike, u = 0.37875
    expected = [(1.5, 0.0), (0.925, 1.0), (0.37875, 1.0)]
    for u_exp, s_exp in expected:
        u, s = lif_step(u, current, V_TH, LEAK)
        assert u.item() == pytest.approx(u_exp)
        assert s.item() == s_exp


def test_soft_reset_subtracts_threshold_once():
    """Far above threshold: one spike, and the potential keeps the excess (no reset to 0)."""
    u, s = lif_step(torch.zeros(1), torch.full((1,), 5.0), V_TH, LEAK)
    assert s.item() == 1.0
    assert u.item() == pytest.approx(3.0)


def test_threshold_is_inclusive():
    assert spike(torch.tensor([V_TH]), V_TH).item() == 1.0
    assert spike(torch.tensor([V_TH - 1e-4]), V_TH).item() == 0.0


def test_no_input_no_spikes():
    u, s = lif_step(torch.zeros(4, 16, 18, 256), torch.zeros(4, 16, 18, 256), V_TH, LEAK)
    assert s.sum() == 0 and u.abs().sum() == 0


@pytest.mark.parametrize("width", [1.0, 2.0])
def test_rectangular_surrogate(width):
    u = torch.tensor([0.5, 1.49, 1.51, 2.0, 2.49, 2.51, 3.5], requires_grad=True)
    spike(u, V_TH, width).sum().backward()
    inside = (u.detach() - V_TH).abs() < width / 2
    expected = inside.float() / width
    torch.testing.assert_close(u.grad, expected)
    if width == 1.0:  # spot-check the boundaries of the default window
        assert u.grad.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_gradient_flows_to_current_and_state():
    u0 = torch.full((3,), 0.5, requires_grad=True)
    current = torch.tensor([0.1, 1.6, 4.0], requires_grad=True)
    u, s = lif_step(u0, current, V_TH, LEAK)
    (u.sum() + s.sum()).backward()
    assert torch.isfinite(current.grad).all() and current.grad.abs().sum() > 0
    assert torch.isfinite(u0.grad).all() and u0.grad.abs().sum() > 0


def test_shape_and_dtype_preserved():
    current = torch.randn(2, 16, 18, 256, dtype=torch.float64)
    u, s = lif_step(torch.zeros_like(current), current, V_TH, LEAK)
    assert u.shape == s.shape == current.shape
    assert u.dtype == s.dtype == torch.float64
    assert set(s.unique().tolist()) <= {0.0, 1.0}
