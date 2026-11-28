"""Theoretical synaptic operation count and energy per window, as in Zhang et al. (2024),
Sec. 4.2.4 / Table 6, with the 45 nm CMOS costs of Horowitz (2014).

* Every synaptic operation of an ANN is a multiply-accumulate (MAC, 4.6 pJ, 32-bit float).
* In an SNN, a module whose input is a binary spike map only accumulates when a spike
  arrives (AC, 0.9 pJ). Its cost is (dense MACs) x (fraction of nonzero inputs).
* Modules with real-valued input stay MACs even in the SNN: the first convolution (it sees
  raw EEG), the readout (it sees firing rates), and any front end. The paper does not
  separate these; here they are counted honestly.

Dense counts come from torch's FLOP counter (FLOPs = 2 x MACs), which also catches
functional convolutions (e.g. braindecode's CombinedConv). Only convolutions and matrix
products are counted; batch norm (foldable into the convolution), pooling, nonlinearities
and LIF state updates are not. Memory access is ignored. This is an estimate for comparing
models, not a measurement of any hardware.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

MAC_PJ = 4.6
AC_PJ = 0.9
SCOPE = ("convolutions and matrix products per window; spike-driven inputs counted as accumulates scaled by "
         "measured input density, real-valued inputs as MACs; excludes batch norm, pooling, LIF updates and "
         "memory access; 45 nm, 32-bit float costs (Horowitz 2014)")


def dense_macs(model: nn.Module, input_shape) -> dict[str, float]:
    """MACs of one window for the whole model (key "") and for each named submodule,
    summed over all calls (e.g. a module run at every SNN time step)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    if hasattr(model, "reset_spike_stats"):
        model.reset_spike_stats()
    prefix = type(model).__name__
    out = {}
    for key, ops in counter.get_flop_counts().items():
        if key == "Global":
            continue
        name = "" if key == prefix else key[len(prefix) + 1:]
        out[name] = sum(ops.values()) / 2
    out.setdefault("", 0.0)
    return out


class SpikeDensityProbe:
    """Records the fraction of nonzero (spiking) inputs to each spike-driven module while
    the model runs. Call ``reset()`` between conditions and ``remove()`` when done."""

    def __init__(self, model: nn.Module):
        self.names = list(model.spike_driven_modules())
        self._handles = [model.register_forward_pre_hook(self._count_windows)]
        self._handles += [module.register_forward_pre_hook(partial(self._record, name))
                          for name, module in model.spike_driven_modules().items()]
        self.reset()

    def reset(self):
        self.windows = 0
        self._density = {name: 0.0 for name in self.names}
        self._calls = {name: 0 for name in self.names}

    def _count_windows(self, module, args):
        self.windows += args[0].shape[0]

    def _record(self, name, module, args):
        x = args[0].detach()
        self._density[name] = self._density[name] + x.flatten(1).ne(0).float().mean(1).sum()
        self._calls[name] += x.shape[0]

    def densities(self) -> dict[str, float]:
        """Mean nonzero-input fraction per call of each spike-driven module."""
        return {name: float(self._density[name]) / self._calls[name] if self._calls[name] else 0.0
                for name in self.names}

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def energy_estimate(dense: dict[str, float], densities: dict[str, float] | None = None) -> dict:
    """Per-window operations and energy. ``densities`` (from ``SpikeDensityProbe``) lists the
    spike-driven modules; without it every operation is a MAC (an ANN, or the SNN
    architecture run densely -- the paper's "RNN" row)."""
    densities = densities or {}
    total = dense[""]
    spiking = [dict(module=name, dense_macs=dense.get(name, 0.0), input_density=rate,
                    accumulates=dense.get(name, 0.0) * rate) for name, rate in densities.items()]
    accumulates = sum(layer["accumulates"] for layer in spiking)
    macs = total - sum(layer["dense_macs"] for layer in spiking)
    return dict(macs=macs, accumulates=accumulates, synaptic_operations=macs + accumulates,
                energy_mj=(macs * MAC_PJ + accumulates * AC_PJ) * 1e-9,
                dense_macs=total, dense_energy_mj=total * MAC_PJ * 1e-9,
                spike_driven=spiking, pj_per_mac=MAC_PJ, pj_per_ac=AC_PJ, scope=SCOPE)


def weighted_firing_rate(rates, neurons) -> float:
    """Firing rate over all hidden neurons (each layer weighted by its neuron count),
    i.e. total spikes / (all neurons x time steps), as in the paper's Table 5/6 average."""
    return float(sum(r * n for r, n in zip(rates, neurons)) / sum(neurons))
