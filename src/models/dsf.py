"""Log-variance dynamic spatial filtering from Banville et al. (2022)."""

import torch
from torch import nn


class DynamicSpatialFilter(nn.Module):
    """Predict a square spatial filter and bias for each EEG window."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.predict_filters = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels * (n_channels + 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Masked channels have zero variance. Replace it before taking the log
        # so both the feature and its gradient stay finite.
        variances = x.var(dim=-1, correction=1)
        log_variances = torch.where(variances > 0, variances, 1.0).log()
        bias, weights = self.predict_filters(log_variances).split(
            [self.n_channels, self.n_channels ** 2], dim=-1
        )
        weights = weights.reshape(-1, self.n_channels, self.n_channels)
        weights = weights.sign() * (weights.abs() - 0.1).clamp_min(0)
        return weights @ x + bias.unsqueeze(-1)
