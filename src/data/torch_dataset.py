"""Window tensors with bounded legacy caching or continuous .npy memory maps."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, patient_ids: list[str],
                 normalization_path: str | Path | None = None, *,
                 normalize: bool = True, return_metadata: bool = False,
                 cache_bytes: int = 512 * 1024**2, n_channels: int = 18,
                 n_times: int = 1024) -> None:
        self.metadata_path = Path(metadata_csv).resolve()
        self.df = pd.read_csv(self.metadata_path)
        self.df = self.df[self.df["patient_id"].isin(patient_ids)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No windows found for patients: {patient_ids}")
        if not self.df["label"].isin([0, 1]).all():
            raise ValueError("Window labels must be binary")
        self.continuous = "continuous_path" in self.df.columns
        path_column = self.df["continuous_path" if self.continuous else "npz_path"]
        paths = {value: self._path(value) for value in path_column.unique()}
        self.paths = path_column.map(paths).to_numpy()
        self.offsets = self.df["start_sample" if self.continuous else "window_index"].to_numpy()
        self.labels = self.df["label"].to_numpy()
        self.shape = (n_channels, n_times)
        self.return_metadata = return_metadata
        if return_metadata and not {"file_id", "start_s", "end_s"}.issubset(self.df.columns):
            raise ValueError("Evaluation requires file_id, start_s and end_s")
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.cache_bytes = cache_bytes
        self._cache = OrderedDict()
        self._bytes = 0
        self.mean = self.std = None
        manifest_path = self.metadata_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and manifest["status"] != "complete":
            raise ValueError("Dataset preparation has not completed successfully")
        if normalize:
            if normalization_path is None:
                raise ValueError("Provide training normalization_path or explicitly set normalize=False")
            with np.load(normalization_path, allow_pickle=False) as norm:
                self.mean = norm["mean"].astype(np.float32)
                self.std = norm["std"].astype(np.float32)
                if manifest is not None:
                    if manifest.get("protocol") and (
                        "protocol" not in norm or str(norm["protocol"]) != manifest["protocol"]
                    ):
                        raise ValueError("Normalization preprocessing protocol differs from dataset manifest")
                    expected_channels = manifest["data_config"]["montage"]["derivations"]
                    if norm["channels"].tolist() != expected_channels:
                        raise ValueError("Normalization channel order differs from dataset manifest")
                    if set(norm["train_patients"].tolist()) != set(manifest["splits"]["train"]):
                        raise ValueError("Normalization training patients differ from dataset manifest")
                    if set(norm["records"].tolist()) != set(manifest["normalization_records"]):
                        raise ValueError("Normalization records differ from dataset manifest")
                    if manifest.get("protocol") == "proposal_unfiltered_v1":
                        if (str(norm["source"]) != "continuous_unfiltered_recordings"
                                or str(norm["protocol"]) != manifest["protocol"]):
                            raise ValueError("Normalization protocol differs from unfiltered proposal dataset")
            if self.mean.shape != (n_channels, 1) or self.std.shape != (n_channels, 1):
                raise ValueError("Normalization must have shape [channels, 1]")
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
                raise ValueError("Normalization must be finite with strictly positive std")
        elif normalization_path is not None:
            raise ValueError("normalize=False conflicts with normalization_path")

    def __len__(self):
        return len(self.df)

    def _path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        # New manifests use paths relative to metadata; legacy paths use project root.
        local = self.metadata_path.parent / path
        return local if local.exists() else path.resolve()

    def _recording(self, path: Path):
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        if path.suffix == ".npy":
            arrays = (np.load(path, mmap_mode="r", allow_pickle=False), None)
            size = 0  # Memory mapped: limit open mappings separately below.
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = (data["X"], data["y"])
            size = sum(a.nbytes for a in arrays)
        while self._cache and (self._bytes + size > self.cache_bytes or len(self._cache) >= 8):
            _, (_, old_size) = self._cache.popitem(last=False)
            self._bytes -= old_size
        if size <= self.cache_bytes:
            self._cache[key] = (arrays, size)
            self._bytes += size
        return arrays

    def __getitem__(self, idx):
        path = self.paths[idx]
        data, labels = self._recording(path)
        offset = int(self.offsets[idx])
        expected_label = int(self.labels[idx])
        if self.continuous:
            x = np.array(data[:, offset:offset + self.shape[1]], dtype=np.float32, copy=True)
            y = expected_label
        else:
            x = np.array(data[offset], dtype=np.float32, copy=True)
            y = int(labels[offset])
        if x.shape != self.shape or not np.isfinite(x).all():
            raise ValueError(f"{path}: invalid EEG window shape or nonfinite values")
        if y not in (0, 1) or y != expected_label:
            raise ValueError(f"{path}: invalid or inconsistent label")
        if self.mean is not None:
            x = (x - self.mean) / self.std
        result = (torch.from_numpy(x), torch.tensor(y, dtype=torch.long))
        if self.return_metadata:
            row = self.df.iloc[idx]
            return (*result, {k: row[k] for k in ("patient_id", "file_id", "start_s", "end_s")})
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _bytes=0)
        return state
