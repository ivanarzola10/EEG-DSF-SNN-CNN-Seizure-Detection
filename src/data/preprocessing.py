"""Causal bandpass filtering.

The bandpass is a single forward pass of an IIR (Butterworth, second-order sections) filter
over each continuous EDF recording. Output at sample n depends only on samples <= n. The
initial filter state is the steady state for a constant input equal to the first sample,
which suppresses the DC step transient while using no future samples.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def design_bandpass(fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    return signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def causal_bandpass(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter a continuous recording ``x`` [C, N] causally; returns float32 [C, N]."""
    x64 = np.asarray(x, dtype=np.float64)
    # sosfilt_zi: (n_sections, 2) -> (n_sections, n_channels, 2)
    zi = signal.sosfilt_zi(sos)[:, None, :] * x64[None, :, 0, None]
    y, _ = signal.sosfilt(sos, x64, axis=-1, zi=zi)
    return y.astype(np.float32)
