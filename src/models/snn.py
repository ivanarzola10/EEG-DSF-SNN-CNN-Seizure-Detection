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

    def neurons_per_layer(self) -> list[int]:
        """LIF neurons in each hidden layer (feature maps x height x width)."""
        widths = [self.input_conv[0].out_channels] + [c[0].out_channels for c in self.hidden_convs]
        return [w * h * t for w, (h, t) in zip(widths, self.layer_shapes)]

    def spike_driven_modules(self) -> dict[str, nn.Module]:
        """Synaptic modules whose input is a binary spike map (relative names). The input
        convolution and the readout receive real values and are not listed."""
        modules = {f"hidden_convs.{i}": conv for i, conv in enumerate(self.hidden_convs)}
        if self.time_steps > 1:
            modules["feedback"] = self.feedback
        return modules

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


class SNNClassifier(nn.Module):
    """Adapter that lets the SNN share the CNN's training and evaluation code.

    Input [B, C, T] -> two class scores [B, 2] = [0, z], where z is the EESNN output logit.
    Softmax of [0, z] equals sigmoid(z), so the seizure probability is still the sigmoid of
    the output potential (proposal Sec. III-B). Class-weighted cross-entropy on [0, z] is
    identical to class-weighted BCE on z, with the same weight-normalized batch mean the CNN
    gets from ``nn.CrossEntropyLoss(weight=...)``.

    ``front_end`` (e.g. DSF) is applied before the first spiking layer and must preserve the
    input shape -- the same contract as ``ShallowEEGCNN``.

    Mean firing rate per hidden layer is accumulated separately for train and eval mode;
    read with ``spike_stats()`` and clear with ``reset_spike_stats()``.
    """

    def __init__(self, model_cfg: dict, n_channels: int = 18, n_times: int = 1024, *,
                 front_end: nn.Module | None = None):
        super().__init__()
        if front_end is not None and not isinstance(front_end, nn.Module):
            raise TypeError("front_end must be a PyTorch nn.Module")
        self.input_shape = (n_channels, n_times)
        self.front_end = front_end
        self.backbone = CompactEESNN(model_cfg, n_channels, n_times)
        self.reset_spike_stats()

    def neurons_per_layer(self) -> list[int]:
        return self.backbone.neurons_per_layer()

    def spike_driven_modules(self) -> dict[str, nn.Module]:
        return {f"backbone.{name}": m for name, m in self.backbone.spike_driven_modules().items()}

    def reset_spike_stats(self) -> None:
        self._spike_sums = {"train": None, "eval": None}
        self._spike_windows = {"train": 0, "eval": 0}

    def spike_stats(self) -> dict:
        """{"train"|"eval": [mean firing rate per hidden layer] or None if no windows seen}."""
        return {mode: None if sums is None else
                [float(s) / self._spike_windows[mode] for s in sums]
                for mode, sums in self._spike_sums.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape}], got {tuple(x.shape)}")
        if self.front_end is not None:
            transformed = self.front_end(x)
            if (not isinstance(transformed, torch.Tensor) or transformed.shape != x.shape
                    or transformed.dtype != x.dtype or transformed.device != x.device):
                raise ValueError("front_end must preserve the input tensor shape, dtype and device")
            x = transformed
        z, rates = self.backbone(x, return_rates=True)
        mode = "train" if self.training else "eval"
        batch = [r.detach().sum() for r in rates]
        sums = self._spike_sums[mode]
        self._spike_sums[mode] = batch if sums is None else [a + b for a, b in zip(sums, batch)]
        self._spike_windows[mode] += len(x)
        return torch.stack([torch.zeros_like(z), z], dim=1)
