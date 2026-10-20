"""Checks against real chb01 recordings; skipped when the EDFs are not downloaded."""
from pathlib import Path

import numpy as np
import pytest

from src.data.annotations import load_summary
from src.data.config import bandpass_from_config, load_config
from src.data.edf_io import load_recording
from src.data.preprocessing import causal_bandpass

CFG = load_config()
RAW = Path(CFG["dataset"]["raw_dir"]) / "chb01"


def _complete(name):
    p = RAW / name
    from src.data.edf_io import read_edf_header
    return p.exists() and p.stat().st_size == read_edf_header(p).expected_file_bytes


pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not (_complete("chb01_01.edf") and _complete("chb01_03.edf")),
                                 reason="chb01 sample EDFs not downloaded")]


def test_summary_matches_physionet_source():
    s = load_summary(RAW / "chb01-summary.txt")
    assert [(x.onset_s, x.offset_s) for x in s["chb01_03"].seizures] == [(2996, 3036)]
    assert s["chb01_01"].seizures == []
    assert sum(len(e.seizures) for e in s.values()) == 7


@pytest.mark.parametrize("name", ["chb01_01.edf", "chb01_03.edf"])
def test_recording_shape(name):
    rec = load_recording(RAW / name, CFG["montage"]["derivations"])
    assert rec.data.shape == (18, 3600 * 256)
    assert rec.extra["duplicates_identical"] == {"T8-P8": True}


def test_seizure_raises_line_length():
    """Sanity check that the annotation lines up with a visible change in the signal."""
    rec = load_recording(RAW / "chb01_03.edf", CFG["montage"]["derivations"])
    x = causal_bandpass(rec.data, bandpass_from_config(CFG))
    ll = np.abs(np.diff(x, axis=1)).mean(0)
    inside = ll[2996 * 256 : 3036 * 256].mean()
    before = ll[2600 * 256 : 2990 * 256].mean()
    assert inside > 1.5 * before
