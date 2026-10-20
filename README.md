# EEG seizure detection under missing channels (DSF × CNN/SNN, CHB-MIT)

Course project studying whether Dynamic Spatial Filtering makes seizure detection more robust
to missing EEG channels, for CNNs vs. spiking networks. See [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)
for the research plan and constraints.

**Status:** data exploration. We can load any CHB-MIT recording, read its seizure annotations,
filter it and plot it. No windowing or models yet.

## The data in one paragraph

[CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) is scalp EEG from 23 children (24 case
folders; `chb21` is the same child as `chb01`), recorded at 256 Hz in roughly 1-hour EDF files:
686 files, ~980 h, 198 annotated seizures. Each channel is a **bipolar derivation**, the voltage
difference between two neighbouring electrodes (e.g. `FP1-F7`), so losing one electrode affects
two channels. We use the 18 derivations listed in [configs/data.yaml](configs/data.yaml). Seizure
start/end times (seconds from file start) come from each folder's `chbXX-summary.txt`.

## Setup (Windows / PowerShell; use `source .venv/bin/activate` elsewhere)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ data\raw   # ~43 GB, git-ignored
```

## Explore

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\audit_chbmit.py                              # -> results/audit/ (usable files, seizures per subject)
& $py scripts\plot_recording.py chb01_03                   # 60 s around the first seizure
& $py scripts\plot_recording.py chb01_01 --start 0 --end 30
& $py -m pytest -q
```

Plots go to `results/figures/`. For interactive work, everything is a plain function:

```python
from src.data.config import load_config, bandpass_from_config
from src.data.edf_io import load_recording
from src.data.annotations import load_summary
from src.data.preprocessing import causal_bandpass

cfg = load_config()
rec = load_recording("data/raw/chb01/chb01_03.edf", cfg["montage"]["derivations"])
rec.data.shape                                    # (18, 921600): channels x samples, µV
load_summary("data/raw/chb01/chb01-summary.txt")["chb01_03"].seizures   # onset/offset in s
x = causal_bandpass(rec.data, bandpass_from_config(cfg))               # 1–40 Hz
```

## What the audit found

673 of 686 files have all 18 derivations (970.6 h, 185 of 198 seizures). Excluded: `chb12_27–29`
(different montage, 13 seizures) and 10 seizure-free `chb24` files with no summary entry.
`chb09_08` has a typo in its summary ("Seizure 1 End" for seizure 2); the parser pairs each
end with the preceding start, so it is read correctly.

## Data decisions so far

All values live in [configs/data.yaml](configs/data.yaml).

| Step | Choice | Why |
|---|---|---|
| Channel labels | Read from the raw EDF header, matched after `strip().upper()` only | MNE renames duplicates (`T8-P8-0`); exact labels are needed for the audit |
| Montage | 18 bipolar derivations; a file missing any one is **excluded** | No silent re-referencing or synthesis |
| Duplicates | First occurrence used; loader checks copies are sample-identical | `T8-P8` appears twice in many files |
| Filter | 4th-order Butterworth 1–40 Hz, forward-only (causal), per EDF file | A real-time detector cannot see the future; files have gaps, so they are not concatenated |
| Subjects | `chb21` = `chb01`; splits will use `subject_id` | Prevents subject leakage |

## Layout

```
configs/data.yaml   dataset paths, 18-derivation montage, filter settings
src/data/           edf_io (load EDFs), annotations (seizure times), montage, preprocessing (filter), config
scripts/            audit_chbmit.py (dataset audit), plot_recording.py (look at a recording)
tests/              unit tests (+ chb01 real-data checks, skipped if the EDFs are absent)
```
