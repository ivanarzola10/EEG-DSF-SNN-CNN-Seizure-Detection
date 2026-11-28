"""Dynamic Spatial Filtering (DSF), Banville et al. 2022, NeuroImage 251:118994.

Ported from ``DynamicSpatialFilter`` in github.com/hubertjb/dynamic-spatial-filtering
(models.py, BSD-3-Clause) for 3D input [B, C, T]. The same module is placed in front of the
CNN and the SNN (``front_end``), keeping C' = C so both backbones see the same shape:

    Phi(X)       spatial summary of the window (no gradient; X is the input)
    h(Phi)       2-layer MLP, C' * (C + 1) outputs: first C' are the bias b, the rest are
                 W reshaped to [C', C] (same slicing order as the reference)
    W'           optional soft threshold sign(W) * max(|W| - tau, 0), tau = 0.1
    m_DSF(X)  =  W' X + b

Phi is either the log-variance of each channel ("logvar", the paper's DSFd) or the upper
triangle of the matrix logarithm of the spatial covariance ("logm", DSFm), computed by
eigendecomposition in float64 on the CPU (the reference also computes logm on the CPU).

Flat channels. Our channel loss writes zeros after z-scoring, so a missing channel has zero
variance and zero covariance eigenvalues. ``flat_log`` sets what log(0) becomes:
  * "clamp" (default): log(max(v, 1e-10)) = -23, as the reference does for logm eigenvalues.
    A missing channel stands out from a z-scored channel, whose log-variance is near 0.
  * "zero": replace log(0) by 0, as in the paper's footnote 3 and the reference logvar code.
    With z-scored inputs this makes a missing channel look like a typical one to DSFd.
"""
from __future__ import annotations

import torch
from torch import nn

LOG_FLOOR = 1e-10  # reference: torch.clamp(e, min=1e-10) before log
SOFT_THRESHOLD = 0.1  # reference and paper Eq. (3)

# Named variants from the paper. ``hidden`` defaults to C**2 (paper Sec. 3.2).
PRESETS = {
    "dsfd": dict(features="logvar", soft_threshold=None),
    "dsfm_st": dict(features="logm", soft_threshold=SOFT_THRESHOLD),
}


def dsf_spec(variant: str | None, n_channels: int = 18) -> dict | None:
    """Resolved settings for a named variant, stored in checkpoint configs; None = no DSF."""
    if variant in (None, "none"):
        return None
    if variant not in PRESETS:
        raise ValueError(f"Unknown DSF variant {variant!r}; choose from {['none', *PRESETS]}")
    return dict(variant=variant, **PRESETS[variant], hidden=n_channels ** 2, flat_log="clamp")


def _log(values: torch.Tensor, flat_log: str) -> torch.Tensor:
    if flat_log == "clamp":
        return values.clamp_min(LOG_FLOOR).log()
    flat = values <= LOG_FLOOR
    return torch.where(flat, torch.zeros_like(values), values.clamp_min(LOG_FLOOR).log())


def spatial_features(x: torch.Tensor, features: str, flat_log: str = "clamp") -> torch.Tensor:
    """Phi(X) for x [B, C, T]: [B, C] ("logvar") or [B, C(C+1)/2] ("logm")."""
    with torch.no_grad():
        xm = x - x.mean(-1, keepdim=True)
        if features == "logvar":
            return _log(xm.pow(2).sum(-1) / (x.shape[-1] - 1), flat_log)
        if features != "logm":
            raise ValueError(f"Unknown DSF features {features!r}")
        cov = (xm @ xm.transpose(1, 2) / (x.shape[-1] - 1)).to("cpu", torch.float64)
        eigenvalues, eigenvectors = torch.linalg.eigh((cov + cov.transpose(1, 2)) / 2)
        logm = eigenvectors @ torch.diag_embed(_log(eigenvalues, flat_log)) @ eigenvectors.transpose(1, 2)
        rows, cols = torch.triu_indices(x.shape[1], x.shape[1])
        return logm[:, rows, cols].to(device=x.device, dtype=x.dtype)


def soft_threshold(w: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(w) * torch.relu(w.abs() - tau)


def channel_importance(w: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Effective channel importance (paper Eq. 4-5): phi_j = ||W[:, j]||_2, [B, C', C] -> [B, C].
    Normalized by each window's maximum, so values are in [0, 1]."""
    phi = w.norm(dim=1)
    return phi / phi.amax(1, keepdim=True).clamp_min(1e-12) if normalize else phi


class DynamicSpatialFilter(nn.Module):
    def __init__(self, n_channels: int = 18, features: str = "logvar", hidden: int | None = None,
                 n_out_channels: int | None = None, soft_threshold: float | None = None,
                 flat_log: str = "clamp", variant: str | None = None):
        super().__init__()
        if features not in ("logvar", "logm"):
            raise ValueError(f"Unknown DSF features {features!r}")
        if flat_log not in ("clamp", "zero"):
            raise ValueError("flat_log must be 'clamp' or 'zero'")
        self.n_channels = n_channels
        self.n_out_channels = n_out_channels or n_channels
        self.features = features
        self.flat_log = flat_log
        self.tau = soft_threshold
        n_in = n_channels if features == "logvar" else n_channels * (n_channels + 1) // 2
        hidden = hidden or n_channels ** 2
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.n_out_channels * (n_channels + 1)))

    @classmethod
    def from_spec(cls, spec: dict, n_channels: int) -> "DynamicSpatialFilter":
        return cls(n_channels, **{k: v for k, v in spec.items() if k != "variant"})

    def filters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatial filters W [B, C', C] (after any soft threshold) and bias b [B, C', 1]."""
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected [batch, {self.n_channels}, time], got {tuple(x.shape)}")
        out = self.mlp(spatial_features(x, self.features, self.flat_log))
        c_out = self.n_out_channels
        bias = out[:, :c_out].unsqueeze(-1)
        w = out[:, c_out:].reshape(len(x), c_out, self.n_channels)
        if self.tau is not None:
            w = soft_threshold(w, self.tau)
        return w, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, bias = self.filters(x)
        return w @ x + bias


class ImportanceProbe:
    """Mean normalized channel importance per input channel while the model runs, split by
    whether the channel was available or masked (an all-zero row) in that window. Recomputes
    the filters in a forward hook, so use it for evaluation only. ``reset()`` between
    conditions, ``remove()`` when done."""

    def __init__(self, dsf: DynamicSpatialFilter):
        self._handle = dsf.register_forward_hook(self._record)
        self.reset()

    def reset(self):
        zeros = torch.zeros(0)
        self._sums = {"available": zeros, "missing": zeros}
        self._counts = {"available": zeros, "missing": zeros}

    def _record(self, module, args, _output):
        x = args[0].detach()
        phi = channel_importance(module.filters(x)[0]).detach().double().cpu()
        missing = (x.abs().amax(-1) == 0).cpu()
        for key, where in (("available", ~missing), ("missing", missing)):
            total, count = (phi * where).sum(0), where.sum(0).double()
            if self._sums[key].numel():
                total, count = total + self._sums[key], count + self._counts[key]
            self._sums[key], self._counts[key] = total, count

    def summary(self) -> dict:
        """{"available"|"missing": per-channel mean (None where never seen), and
        "<key>_mean": mean over all (window, channel) pairs of that kind}."""
        out = {}
        for key in ("available", "missing"):
            total, count = self._sums[key], self._counts[key]
            out[key] = [float(s / c) if c else None for s, c in zip(total, count)]
            out[f"{key}_mean"] = float(total.sum() / count.sum()) if count.numel() and count.sum() else None
        return out

    def remove(self):
        self._handle.remove()
