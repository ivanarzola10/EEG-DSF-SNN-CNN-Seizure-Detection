"""Compact EESNN: a recurrent convolutional spiking network (Zhang et al. 2024), reduced in
depth and width. Hyperparameters come from ``configs/snn.yaml``.

The window [B, C, T] enters as a one-channel image [B, 1, C, T]; every convolution is 2D
over (EEG channel, time). For SNN time steps t = 0..time_steps-1 the same input is
presented again, and

    layer 1:  I1[t] = BN(F1 x) + W1 s_H[t-1]      (feedback term absent at t = 0)
    layer l:  Il[t] = BN(Fl s_{l-1}[t])           l = 2..H

each followed by an LIF update (``src.models.lif``). With one hidden layer, W1 feeds layer 1
back into itself. The readout is the leak-weighted average firing rate of layer H,

    r = sum_t leak^(T-1-t) s_H[t] / sum_t leak^t,

average-pooled over time, flattened, and mapped to one logit by a Linear layer. Apply a
sigmoid (or BCEWithLogitsLoss) to get the seizure probability.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .lif import lif_step

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "snn.yaml"


def load_snn_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _same_padding(kernel: tuple[int, int]) -> tuple[int, int]:
    return tuple((k - 1) // 2 for k in kernel)


def _conv_out(size: tuple[int, int], kernel, stride, padding) -> tuple[int, int]:
    return tuple((n + 2 * p - k) // s + 1 for n, k, s, p in zip(size, kernel, stride, padding))


class CompactEESNN(nn.Module):
    def __init__(self, model_cfg: dict, n_channels: int = 18, n_samples: int = 1024):
        super().__init__()
        cfg = model_cfg
        if cfg["surrogate"] != "rectangular":
            raise ValueError(f"unsupported surrogate {cfg['surrogate']!r}")
        if cfg["dropout"] != 0:
            # The reference uses a dropout mask held fixed across time steps; not implemented.
            raise ValueError("dropout > 0 is not implemented")
        n_hidden = cfg["n_hidden"]
        if not 1 <= n_hidden <= len(cfg["channels"]):
            raise ValueError(f"n_hidden={n_hidden} needs that many entries in channels")
        widths = cfg["channels"][:n_hidden]

        self.time_steps = cfg["time_steps"]
        self.v_th = float(cfg["v_th"])
        self.leak = float(cfg["leak"])
        self.width = float(cfg["surrogate_width"])

        def conv_bn(c_in, c_out, kernel, stride):
            pad = _same_padding(kernel)
            layers = [nn.Conv2d(c_in, c_out, kernel, stride, pad)]
            if cfg["batch_norm"]:
                layers.append(nn.BatchNorm2d(c_out))
            return nn.Sequential(*layers), pad

        k_in, s_in = tuple(cfg["input_kernel"]), tuple(cfg["input_stride"])
        k_h, s_h = tuple(cfg["hidden_kernel"]), tuple(cfg["hidden_stride"])

        self.input_conv, pad = conv_bn(1, widths[0], k_in, s_in)
        shape = _conv_out((n_channels, n_samples), k_in, s_in, pad)
        self.layer_shapes = [shape]
        self.hidden_convs = nn.ModuleList()
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            conv, pad = conv_bn(c_in, c_out, k_h, s_h)
            self.hidden_convs.append(conv)
            shape = _conv_out(shape, k_h, s_h, pad)
            self.layer_shapes.append(shape)

        # W1: last hidden layer -> layer 1. A transposed conv undoes the hidden strides;
        # output_size is passed at call time so the shape always matches layer 1.
        k_fb = tuple(cfg["feedback_kernel"])
        fb_stride = tuple(s ** (n_hidden - 1) for s in s_h)
        self.feedback = nn.ConvTranspose2d(
            widths[-1], widths[0], k_fb, fb_stride, _same_padding(k_fb), bias=False
        )
        if cfg["feedback_spectral_norm"]:
            self.feedback = nn.utils.parametrizations.spectral_norm(self.feedback)

        pool_t = min(cfg["readout_pool_time"], shape[1])
        self.readout_pool = nn.AdaptiveAvgPool2d((shape[0], pool_t))
        self.readout = nn.Linear(widths[-1] * shape[0] * pool_t, 1)

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG, **kwargs) -> "CompactEESNN":
        return cls(load_snn_config(path)["model"], **kwargs)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, return_rates: bool = False):
        """``x`` [B, C, T] -> logits [B]. With ``return_rates``, also a list with each hidden
        layer's firing rate per window ([B], spikes / (neurons * time_steps))."""
        current_in = self.input_conv(x.unsqueeze(1))  # constant input: computed once
        u = [None] * (len(self.hidden_convs) + 1)
        spike_sums = [0.0] * len(u)
        trace = 0.0
        s = None
        for t in range(self.time_steps):
            current = current_in
            if t > 0:
                current = current + self.feedback(s, output_size=current_in.shape[-2:])
            for i in range(len(u)):
                if i > 0:
                    current = self.hidden_convs[i - 1](s)
                if u[i] is None:
                    u[i] = torch.zeros_like(current)
                u[i], s = lif_step(u[i], current, self.v_th, self.leak, self.width)
                if return_rates:
                    spike_sums[i] = spike_sums[i] + s.detach().flatten(1).mean(1)
            trace = self.leak * trace + s

        norm = sum(self.leak ** t for t in range(self.time_steps))
        rate = self.readout_pool(trace / norm)
        logits = self.readout(rate.flatten(1)).squeeze(-1)
        if return_rates:
            return logits, [r / self.time_steps for r in spike_sums]
        return logits
