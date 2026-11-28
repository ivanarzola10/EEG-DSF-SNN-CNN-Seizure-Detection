"""Proposal CNN: Braindecode ShallowFBCSPNet with window-end predictions."""
from __future__ import annotations

import torch
from torch import nn
from braindecode.models import ShallowFBCSPNet

from src.models.dsf import DynamicSpatialFilter


class ShallowEEGCNN(nn.Module):
    """Accept [batch, channels, samples], return two unnormalized class scores.

    A prediction uses the complete supplied window and is timestamped at its end.
    The constructor exposes temporal-filter and pooling settings for validation.
    """

    def __init__(self, n_channels=18, n_classes=2, n_temporal_filters=40,
                 temporal_kernel_size=25, pool_size=75, pool_stride=15,
                 dropout=0.5, n_times=1024, *, front_end: nn.Module | None = None,
                 dsf: bool = False):
        super().__init__()
        self.input_shape = (n_channels, n_times)
        if front_end is not None and not isinstance(front_end, nn.Module):
            raise TypeError("front_end must be a PyTorch nn.Module")
        if dsf and front_end is not None:
            raise ValueError("Choose either the built-in DSF or a custom front end")
        self.backbone = ShallowFBCSPNet(
            n_chans=n_channels, n_outputs=n_classes, n_times=n_times,
            n_filters_time=n_temporal_filters, n_filters_spat=n_temporal_filters,
            filter_time_length=temporal_kernel_size,
            pool_time_length=pool_size, pool_time_stride=pool_stride,
            final_conv_length="auto", pool_mode="mean", drop_prob=dropout,
        )
        self.front_end = DynamicSpatialFilter(n_channels) if dsf else front_end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape}], got {tuple(x.shape)}")
        if self.front_end is not None:
            transformed = self.front_end(x)
            if (not isinstance(transformed, torch.Tensor) or transformed.shape != x.shape
                    or transformed.dtype != x.dtype or transformed.device != x.device):
                raise ValueError("front_end must preserve the input tensor shape, dtype and device")
            x = transformed
        return self.backbone(x)
