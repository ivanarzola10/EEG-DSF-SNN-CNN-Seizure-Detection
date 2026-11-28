import pytest
import torch
from torch import nn

from src.evaluation.energy import (AC_PJ, MAC_PJ, SpikeDensityProbe, dense_macs, energy_estimate,
                                   weighted_firing_rate)
from src.models.snn import SNNClassifier, load_snn_config


class TinyANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, (3, 5), padding=0)
        self.fc = nn.Linear(4 * 2 * 6, 2)

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1))


def test_dense_macs_match_hand_count():
    dense = dense_macs(TinyANN(), (4, 10))
    conv = 4 * 2 * 6 * (1 * 3 * 5)  # output elements x (input channels x kernel)
    fc = 48 * 2
    assert dense["conv"] == conv and dense["fc"] == fc and dense[""] == conv + fc
    energy = energy_estimate(dense)
    assert energy["accumulates"] == 0 and energy["macs"] == conv + fc
    assert energy["energy_mj"] == pytest.approx((conv + fc) * MAC_PJ * 1e-9)


def test_snn_spike_driven_modules_cost_accumulates_scaled_by_density():
    cfg = {**load_snn_config()["model"], "n_hidden": 2}
    model = SNNClassifier(cfg).eval()
    dense = dense_macs(model, model.input_shape)
    assert model.spike_stats() == {"train": None, "eval": None}  # counting left no trace
    names = set(model.spike_driven_modules())
    assert names == {"backbone.hidden_convs.0", "backbone.feedback"}
    probe = SpikeDensityProbe(model)
    with torch.no_grad():
        model(torch.randn(3, *model.input_shape) * 5)
    densities = probe.densities()
    assert probe.windows == 3 and all(0 <= d <= 1 for d in densities.values())
    energy = energy_estimate(dense, densities)
    spike_dense = sum(dense[n] for n in names)
    assert energy["macs"] == pytest.approx(dense[""] - spike_dense)
    assert energy["accumulates"] == pytest.approx(sum(dense[n] * densities[n] for n in names))
    assert energy["dense_macs"] == dense[""]
    expected = (energy["macs"] * MAC_PJ + energy["accumulates"] * AC_PJ) * 1e-9
    assert energy["energy_mj"] == pytest.approx(expected)
    # Silent network: spike-driven modules cost nothing.
    assert energy_estimate(dense, {n: 0.0 for n in names})["accumulates"] == 0
    probe.remove()
    model(torch.randn(1, *model.input_shape))
    assert probe.windows == 3  # hooks are gone


def test_weighted_firing_rate_weights_layers_by_neuron_count():
    assert weighted_firing_rate([0.1, 0.4], [300, 100]) == pytest.approx(0.175)
    cfg = {**load_snn_config()["model"], "n_hidden": 2}
    neurons = SNNClassifier(cfg).neurons_per_layer()
    shapes = SNNClassifier(cfg).backbone.layer_shapes
    assert neurons == [cfg["channels"][i] * shapes[i][0] * shapes[i][1] for i in range(2)]
