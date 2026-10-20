import numpy as np
import pytest

from src.data.preprocessing import causal_bandpass, design_bandpass

FS = 256.0


@pytest.fixture
def sos():
    return design_bandpass(FS, 1.0, 40.0, 4)


def test_filter_is_causal(sos):
    """Changing future samples must not change any earlier output sample."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5000))
    y = causal_bandpass(x, sos)
    x_future = x.copy()
    x_future[:, 3000:] += rng.normal(scale=100, size=(3, 2000))
    y_future = causal_bandpass(x_future, sos)
    np.testing.assert_array_equal(y[:, :3000], y_future[:, :3000])
    assert not np.allclose(y[:, 3000:], y_future[:, 3000:])


def test_dc_offset_has_no_start_transient(sos):
    x = np.full((1, 2048), 1000.0)
    assert np.abs(causal_bandpass(x, sos)).max() < 1e-6


def test_passband_and_stopband(sos):
    t = np.arange(int(20 * FS)) / FS
    settle = int(5 * FS)
    for f, passes in [(10.0, True), (0.1, False), (100.0, False)]:
        y = causal_bandpass(np.sin(2 * np.pi * f * t)[None], sos)[0, settle:]
        gain = y.std() / np.sin(2 * np.pi * f * t[settle:]).std()
        assert (gain > 0.9) if passes else (gain < 0.05), (f, gain)
