"""Exercise real EEG when the corresponding frozen-cohort EDFs are available."""
from pathlib import Path

import mne
import numpy as np
import pytest

from scripts.prepare_dataset import Source, window_rows
from src.data.annotations import load_summary
from src.data.config import load_config
from src.data.edf_io import load_recording, read_edf_header

CFG = load_config()
RAW = Path(CFG["dataset"]["raw_dir"]) / "chb01"
pytestmark = pytest.mark.data


def complete_edf(file_id):
    path = RAW / f"{file_id}.edf"
    if not path.exists() or path.stat().st_size != read_edf_header(path).expected_file_bytes:
        pytest.skip(f"{path.name} is unavailable or incomplete")
    return path


@pytest.mark.parametrize("file_id", ["chb01_01", "chb01_15"])
def test_ordered_real_recording(file_id):
    path = complete_edf(file_id)
    channels = CFG["montage"]["derivations"]
    rec = load_recording(path, channels)
    assert rec.data.shape == (18, 3600 * 256)
    assert rec.channels == tuple(channels)
    assert rec.extra["duplicates_identical"] == {"T8-P8": True}
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    picks = [rec.header.labels.index(channel) for channel in channels]
    expected = (raw.get_data(picks=picks, start=800, stop=1824) * 1e6).astype(np.float32)
    np.testing.assert_array_equal(rec.data[:, 800:1824], expected)


def test_real_summary_and_window_boundaries():
    path = complete_edf("chb01_15")
    entries = load_summary(RAW / "chb01-summary.txt")
    (event,) = entries[path.stem].seizures
    assert (event.onset_s, event.offset_s) == (1732, 1772)
    assert entries["chb01_01"].seizures == []
    source = Source(path, "chb01", "train", 3600 * 256, (event,))
    windows = list(window_rows(source, CFG))
    assert windows[0]["start_s"] == 0
    assert windows[-1]["end_s"] == 3600
    assert windows[1729]["label"] == 0  # Ends one second after onset.
    assert windows[1730]["label"] == 1  # Ends two seconds after onset.
    assert windows[1730]["end_s"] == 1734
