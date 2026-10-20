"""EDF header parsing and signal loading.

The header parser is a dependency-free reader of the EDF fixed-width header. It returns the
*exact* channel labels stored in the file (MNE renames duplicates such as ``T8-P8`` to
``T8-P8-0``/``T8-P8-1``).

Signal data are read with MNE and picked by header index, so label renaming cannot cause a
wrong channel to be selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .montage import MontageMatch, match_montage

FIXED_HEADER_BYTES = 256
BYTES_PER_SIGNAL_HEADER = 256


@dataclass(frozen=True)
class EdfHeader:
    n_signals: int
    n_records: int
    record_duration_s: float
    labels: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    physical_dims: tuple[str, ...]
    start_date: str
    start_time: str

    @property
    def sampling_rates(self) -> tuple[float, ...]:
        return tuple(n / self.record_duration_s for n in self.samples_per_record)

    @property
    def duration_s(self) -> float:
        return self.n_records * self.record_duration_s

    @property
    def header_bytes(self) -> int:
        return FIXED_HEADER_BYTES + self.n_signals * BYTES_PER_SIGNAL_HEADER

    @property
    def expected_file_bytes(self) -> int:
        """EDF stores 16-bit samples, so a complete file has exactly this many bytes."""
        return self.header_bytes + 2 * self.n_records * sum(self.samples_per_record)


def parse_edf_header(raw: bytes) -> EdfHeader:
    """Parse an EDF header from its leading bytes (at least 256 * (1 + n_signals))."""
    if len(raw) < FIXED_HEADER_BYTES:
        raise ValueError(f"EDF header needs >= {FIXED_HEADER_BYTES} bytes, got {len(raw)}")

    def field_str(start: int, width: int) -> str:
        return raw[start : start + width].decode("ascii", errors="replace").strip()

    n_signals = int(field_str(252, 4))
    needed = FIXED_HEADER_BYTES + n_signals * BYTES_PER_SIGNAL_HEADER
    if len(raw) < needed:
        raise ValueError(f"EDF header with {n_signals} signals needs {needed} bytes, got {len(raw)}")

    # Signal header fields are stored field-by-field: all labels, then all transducers, ...
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    offsets = np.cumsum([FIXED_HEADER_BYTES] + [w * n_signals for w in widths[:-1]])

    def signal_fields(index: int) -> list[str]:
        base, width = int(offsets[index]), widths[index]
        return [field_str(base + i * width, width) for i in range(n_signals)]

    return EdfHeader(
        n_signals=n_signals,
        n_records=int(field_str(236, 8)),
        record_duration_s=float(field_str(244, 8)),
        labels=tuple(signal_fields(0)),
        physical_dims=tuple(signal_fields(2)),
        samples_per_record=tuple(int(s) for s in signal_fields(8)),
        start_date=field_str(168, 8),
        start_time=field_str(176, 8),
    )


def read_edf_header(path: str | Path) -> EdfHeader:
    with open(path, "rb") as f:
        fixed = f.read(FIXED_HEADER_BYTES)
        n_signals = int(fixed[252:256].decode("ascii").strip())
        rest = f.read(n_signals * BYTES_PER_SIGNAL_HEADER)
    return parse_edf_header(fixed + rest)


def file_id_from_path(path: str | Path) -> str:
    """'data/raw/chb01/chb01_03.edf' -> 'chb01_03' (also handles 'chb02_16+')."""
    return Path(path).name.removesuffix(".edf")


def patient_id_from_file_id(file_id: str) -> str:
    """'chb01_03' -> 'chb01'. Note chb21 is the same subject as chb01 (see splits)."""
    return file_id.split("_")[0]


@dataclass
class Recording:
    """One EDF restricted to the verified montage. ``data`` is [channels, samples] in µV."""

    data: np.ndarray
    fs: float
    channels: tuple[str, ...]
    file_id: str
    patient_id: str
    header: EdfHeader
    montage: MontageMatch
    extra: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


def load_recording(
    path: str | Path,
    derivations: list[str],
    expected_fs: float = 256.0,
    check_duplicates: bool = True,
) -> Recording:
    """Load an EDF, verify its montage, and return the derivations in config order.

    Raises ``ValueError`` if any derivation is missing or the sampling rate differs from
    ``expected_fs`` (no resampling or re-referencing is ever done silently).
    """
    import mne  # imported lazily: header-only audits don't need it

    path = Path(path)
    header = read_edf_header(path)
    if path.stat().st_size != header.expected_file_bytes:
        raise ValueError(f"{path.name}: {path.stat().st_size} bytes, header implies "
                         f"{header.expected_file_bytes} (truncated download?)")
    match = match_montage(header.labels, derivations)
    if match.missing:
        raise ValueError(f"{path.name}: missing derivations {match.missing}")
    rates = {header.sampling_rates[i] for i in match.indices}
    if rates != {expected_fs}:
        raise ValueError(f"{path.name}: sampling rate(s) {sorted(rates)} != {expected_fs}")

    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    if len(raw.ch_names) != header.n_signals or raw.info["sfreq"] != expected_fs:
        raise ValueError(
            f"{path.name}: MNE sees {len(raw.ch_names)} ch @ {raw.info['sfreq']} Hz, "
            f"header says {header.n_signals} ch"
        )
    # MNE returns volts; convert to microvolts for readable magnitudes.
    data = (raw.get_data(picks=list(match.indices)) * 1e6).astype(np.float32)

    extra: dict = {}
    if check_duplicates and match.duplicates:
        # Record whether repeated derivations are sample-identical copies.
        extra["duplicates_identical"] = {}
        for name, idxs in match.duplicates.items():
            copies = raw.get_data(picks=list(idxs))
            extra["duplicates_identical"][name] = all(np.array_equal(copies[0], c) for c in copies[1:])

    file_id = file_id_from_path(path)
    return Recording(
        data=data,
        fs=expected_fs,
        channels=tuple(derivations),
        file_id=file_id,
        patient_id=patient_id_from_file_id(file_id),
        header=header,
        montage=match,
        extra=extra,
    )
