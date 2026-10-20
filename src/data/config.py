"""Load ``configs/data.yaml`` and build pipeline objects from it."""
from __future__ import annotations

from pathlib import Path

import yaml

from .preprocessing import design_bandpass

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["raw_dir"] = str(REPO_ROOT / cfg["dataset"]["raw_dir"])
    return cfg


def bandpass_from_config(cfg: dict):
    bp = cfg["preprocessing"]["bandpass"]
    return design_bandpass(cfg["dataset"]["sampling_rate_hz"], bp["low_hz"], bp["high_hz"], bp["order"])
