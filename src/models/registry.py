"""Build a detector from a checkpoint config, so one evaluator serves the CNN and the SNN."""
from __future__ import annotations

from torch import nn


def build_model(config: dict, front_end: nn.Module | None = None) -> nn.Module:
    """``config["model_type"]`` is "cnn" (default, for checkpoints saved before this key
    existed) or "snn"; ``config["model"]`` holds that model's constructor settings.

    ``config["dsf"]`` (from ``src.models.dsf.dsf_spec``; absent or None = no DSF) attaches a
    Dynamic Spatial Filter as the front end. It is created after the backbone, so with the
    same seed the backbone starts from the same weights with and without DSF."""
    kind = config.get("model_type", "cnn")
    if kind == "cnn":
        from .cnn import ShallowEEGCNN  # imported lazily: needs braindecode
        model = ShallowEEGCNN(**config["model"], front_end=front_end)
    elif kind == "snn":
        from .snn import SNNClassifier
        model = SNNClassifier(config["model"], front_end=front_end)
    else:
        raise ValueError(f"Unknown model_type {kind!r}")
    spec = config.get("dsf")
    if spec:
        if front_end is not None:
            raise ValueError("config defines a DSF front end; do not also pass front_end")
        from .dsf import DynamicSpatialFilter
        model.front_end = DynamicSpatialFilter.from_spec(spec, model.input_shape[0])
    return model
