"""Model-independent missing-channel policies, applied after normalization."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch.utils.data import Dataset


def validate_experiment(experiment):
    fractions = experiment["missing_fractions"]
    weights = experiment["training_mask_probabilities"]
    if fractions != [0.0, 0.25, 0.5, 0.75]:
        raise ValueError("Expected the four proposal channel-loss fractions")
    if len(weights) != len(fractions) or any(w < 0 for w in weights) or not np.isclose(sum(weights), 1):
        raise ValueError("Mask probabilities must sum to one")
    if weights[0] <= 0 or any(w <= 0 for w in weights):
        raise ValueError("Training must include every condition, including complete inputs")
    if experiment["evaluation_repeats"] < 1:
        raise ValueError("At least one mask repeat is required")


def training_mask(x, generator, fractions, probabilities):
    """Randomize each window independently; return EEG and availability [B,C]."""
    if x.ndim != 3:
        raise ValueError("Expected [batch, channels, time]")
    batch, channels, _ = x.shape
    levels = torch.multinomial(torch.tensor(probabilities, dtype=torch.float32), batch,
                               replacement=True, generator=generator)
    removed = torch.tensor([math.ceil(channels * f) for f in fractions])[levels]
    ranks = torch.rand(batch, channels, generator=generator).argsort(1).argsort(1)
    available = (ranks >= removed[:, None]).to(device=x.device, dtype=x.dtype)
    return x * available[:, :, None], available


def make_mask_manifest(dataset_manifest, experiment, dataset_fingerprint):
    validate_experiment(experiment)
    channels = dataset_manifest["data_config"]["montage"]["derivations"]
    records = sorted((r["split"], r["file_id"]) for r in dataset_manifest["included"] if r["split"] in ("val", "test"))
    masks = []
    for split, file_id in records:
        for fraction in experiment["missing_fractions"]:
            count = math.ceil(len(channels) * fraction)
            repeats = 1 if fraction == 0 else experiment["evaluation_repeats"]
            if repeats > math.comb(len(channels), count):
                raise ValueError("More mask repeats requested than distinct channel subsets")
            used = set()
            for repeat in range(repeats):
                key = f"{experiment['evaluation_mask_seed']}:{split}:{file_id}:{fraction}:{repeat}"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                rng = np.random.default_rng(seed)
                missing = tuple(sorted(rng.choice(len(channels), count, replace=False).tolist()))
                # Repeats for a recording/rate should represent distinct subsets.
                while missing in used:
                    missing = tuple(sorted(rng.choice(len(channels), count, replace=False).tolist()))
                used.add(missing)
                masks.append(dict(split=split, file_id=file_id, missing_fraction=fraction,
                                  repeat=repeat, missing_indices=list(missing)))
    return dict(version="recording_masks_v1", dataset_manifest_sha256=dataset_fingerprint,
                channels=channels, experiment=experiment, masks=masks)


def validate_mask_manifest(saved, dataset_manifest, experiment, dataset_fingerprint):
    # Exact regeneration checks coverage, channels, subsets and all seed settings.
    expected = make_mask_manifest(dataset_manifest, experiment, dataset_fingerprint)
    if saved != expected:
        raise ValueError("Mask manifest differs from the frozen dataset/experiment policy")


class RecordingMaskedDataset(Dataset):
    """Apply one fixed channel subset to every window of each EDF."""
    def __init__(self, base, mask_manifest, split, fraction, repeat):
        self.base = base
        self.files = base.df.file_id.to_numpy()
        self.availability = {}
        for row in mask_manifest["masks"]:
            if (row["split"], row["missing_fraction"], row["repeat"]) == (split, fraction, repeat):
                keep = torch.ones(len(mask_manifest["channels"]), 1)
                keep[row["missing_indices"]] = 0
                self.availability[row["file_id"]] = keep
        if set(self.files) != set(self.availability):
            raise ValueError("Masks must cover exactly the evaluated recordings")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        return x * self.availability[self.files[index]], y
