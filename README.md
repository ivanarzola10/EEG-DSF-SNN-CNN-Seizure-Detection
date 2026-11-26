# ECE 210 · Standalone EEG seizure CNN

The standalone baseline uses Braindecode ShallowFBCSPNet on patient-disjoint
CHB-MIT recordings. The prepared dataset and recording masks are shared inputs
for the joint experiments. [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) contains the
joint research notes.

## Setup and run

Use Python 3.11 and install `requirements.txt` in `.venv`. Put the CHB-MIT 1.0.0
case folders, EDFs and case summaries in `data/raw/`. The frozen inventory below
lists the required EDFs. It is also possible to put the data elsewhere using
`--raw-dir /path/to/chbmit`. A fresh prepared cache needs about 26 GiB in
addition to the raw data. The complete architecture search and held-out mask
evaluation run over millions of windows.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

From the repository root, with the environment and EDFs available, run:

```bash
.venv/bin/python -m scripts.train_cnn
```

The command validates the raw recording inventory, reuses a compatible cache or
builds a fresh one, trains and selects the CNN using validation data, writes the
validation alarm settings, then reloads the selected checkpoint and scores the
test split. It automatically chooses CUDA, MPS, or CPU, in that order. Use
`--device cpu`, `--device mps`, or `--device cuda` to pin the backend. On Windows
use `.venv\Scripts\python.exe` as the interpreter.

Use the same entry point with `--smoke` for an integration check. It validates
the full inventory and uses one short real EDF excerpt per split, with timestamps
relative to each excerpt. It runs one epoch per reduced candidate and executes
all 16 test mask conditions under `runs/cnn_smoke/<UTC timestamp>/`. Run the
suite with `.venv/bin/python -m pytest -q`.

## Frozen cohort

[cohort.json](configs/cohort.json) is authoritative for the split. The project
ranked locally available subjects with retained seizures and no observed integrity
flags by usable recording coverage, usable recording count, and subject ID.
Three ranked strata of six patients each contributed four train, one validation,
and one test patient using NumPy's generator with seed 42. The seizure counts are
the retained original annotations, independent of model scores.

| Split | Patients (retained seizures) | Patients | EDFs | Seizures |
| --- | --- | ---: | ---: | ---: |
| Train | chb01 + chb21 (7), chb02 (3), chb03 (4), chb05 (3), chb06 (4), chb09 (3), chb10 (5), chb11 (3), chb13 (7), chb16 (8), chb20 (8), chb22 (2) | 12 | 266 | 57 |
| Validation | chb08 (4), chb18 (5), chb19 (2) | 3 | 58 | 11 |
| Test | chb04 (2), chb14 (5), chb23 (5) | 3 | 32 | 12 |

[recording_inventory.csv](configs/recording_inventory.csv) is the frozen audit
inventory. All 356 rows marked `usable` for selected cases are required,
including seizure-free recordings. The inventory records 204 downloads that were
unavailable in those cases when the cohort was fixed (131 train, 28 validation,
45 test). These files stay outside this experiment even if later downloaded.
An absent required EDF or summary raises an error with its path. This is an
availability-limited cohort of 18 unique patients across 19 case folders.

The proposal's five recording exclusions describe dataset-wide findings, and
all five are outside the frozen cohort:

| Recording | Reason |
| --- | --- |
| `chb12_27`, `chb12_28`, `chb12_29` | Historical montage audit found incompatible derivations. The local audit directly confirmed `chb12_28`; the other two were unavailable locally. |
| `chb07_18` | `RECORDS-WITH-SEIZURES` lists this EDF but the summary places the event in `chb07_19`. The audit flags the related mismatch on both rows, and the latter EDF was unavailable locally. |
| `chb17b_69` | Actual EDF size differs from the header-declared size. |

The stricter local audit also records other dataset-wide issues in the inventory.
No recording-level quality exclusions apply within the frozen 18-patient cohort.
For exploratory auditing, `python scripts/audit_chbmit.py` generates fresh
reports under `results/audit/`.

## Prepared dataset

The preferred cache is `data/processed/prepared_proposal/`. Incompatible output
is preserved and the replacement uses a unique sibling directory. The actual
path is printed and saved in the run provenance. Run
`python -m scripts.prepare_dataset` for data preparation and cache validation
alone. The old `prepared/` dataset used a different split and causal filtering.
The canonical CNN uses the proposal's unfiltered signals. The causal bandpass in
[configs/data.yaml](configs/data.yaml) is used by the exploratory plotting script.

The canonical pipeline matches EDF labels after trimming and uppercasing, uses
the first of any sample-identical duplicates, and keeps the 18 derivations in
the order in `configs/data.yaml`. Signals are continuous float32 microvolts at
256 Hz, with no resampling or imputation. Each 4-second window spans 1024
samples with a 256-sample stride. Its label is 1 if at least 2 seconds overlap
the union of seizure intervals. Window timestamps are seconds from EDF start.

| File | Contents |
| --- | --- |
| `recordings/<file_id>.npy` | `[18, samples]` continuous EEG, shared by overlapping windows |
| `metadata.csv` | Case folder (`patient_id`), canonical identity (`subject_id`), EDF ID, split, start sample, start/end seconds, binary label and relative signal path |
| `seizures.csv` | Original interval onset/offset seconds, case, EDF and event IDs |
| `normalization.npz` | Per-channel `[18, 1]` mean and population std, ordered channels, source train cases and EDFs, sample count and protocol |
| `manifest.json` | Protocol, cohort, resolved preprocessing, split, window parameters and recording counts |

`WindowDataset(root / "metadata.csv", manifest["splits"][split],
root / "normalization.npz")` returns float32 EEG `[18, 1024]` and an integer
label. Normalization counts each continuous training sample once and floors
std at 0.0001 microvolts. All splits use those training statistics. Request
`return_metadata=True` to get EDF identity and window timestamps alongside each
sample.

## Training and evaluation

The default sweep compares 20 or 40 temporal and spatial filters, temporal
kernels of 25 or 51 samples, and pooling length/stride of 75/15 or 150/30 samples.
Each candidate uses seed 42, Adam at 0.001, dropout 0.5, gradient clipping at 5,
batch size 16, and up to 30 epochs with patience 5. Training uniformly shuffles
windows without replacement and uses inverse-frequency weighted cross-entropy.
Training masks choose complete, 25%, 50%, or 75% missing with equal probability
per window. Validation and test use all their windows at natural prevalence.
The highest complete-input validation trapezoidal PR-AUC selects the epoch and
architecture, with ties favoring the earlier epoch and candidate. Average
precision is the recall-weighted mean of precision without linear interpolation.
RNG settings are fixed, with deterministic Torch algorithms requested where
supported. Exact scores can vary across hardware and builds.

[experiment.json](configs/experiment.json) defines mask and alarm settings.
Evaluation masks drop 0, 5, 9, or 14 derivations by rounding up. Masking writes
zeros after normalization. Masked conditions use five distinct subsets per EDF
and loss level, fixed throughout the EDF. The saved `evaluation_masks.json`
gives exact missing indices in montage order. The same assignments can be reused
with `RecordingMaskedDataset` on the same manifest.

On complete-input validation recordings, the command selects the score
threshold, consecutive-window requirement, and cooldown to maximize pooled
seizure sensitivity at or below one false alarm per non-seizure hour. Ties favor
lower mean delay, fewer false alarms per hour, then higher threshold. The same
frozen rule applies to every held-out condition.

Each probability becomes available at the window end. Scores strictly above
threshold contribute to the required streak. A streak emits at most one alert
once cooldown allows. The streak breaks on a below-or-equal score or a timestamp
gap, and all alarm state resets at each EDF boundary. The first alert in the
onset-inclusive, offset-exclusive seizure interval detects the event. Delay is
alert time minus onset for detected events. An alert outside all seizure intervals
is false. The denominator is the entire recorded duration minus annotated seizure
time, including startup and excluding gaps between EDFs.

## Run outputs

Each run creates `runs/cnn/<UTC timestamp>/`.

- `best.pt` contains the selected standalone model weights and model configuration.
  `config.json`, `provenance.json`, `dataset_manifest.json`, `normalization.npz`
  and `selection.json` capture the resolved experiment and validation selection.
- `validation/calibration.json` records the frozen rule before held-out inference.
  `evaluation_masks.json` contains the shared recording mask assignments.
- `test/patient_metrics.csv` contains patient and subset results with paired
  changes from complete input (masked minus complete). `test/pooled_metrics.csv`
  pools the patients within each subset. `test/summary.json` summarizes patient
  and subset variability and stores parameter count and hardware, device, batch
  size and forward-only timing. Patient summaries omit undefined metrics.
- `test/seizure_events.csv` lists detections, delays and missed seizures.
  `test/alerts.csv` lists alert timestamps and false-alarm status.
  `test/predictions/*.npz` follows the test rows in cache `metadata.csv`, in order.

## Optional front end

`ShallowEEGCNN` wraps one Braindecode 1.2.0 backbone. It accepts
`[batch, 18, 1024]` and returns two raw scores per window for
`CrossEntropyLoss`. The seizure probability is `logits.softmax(1)[:, 1]`.

Supply an initialized `torch.nn.Module` via `front_end`. It receives normalized
and already masked EEG in canonical montage order immediately before the
backbone. Return a tensor of the same shape, dtype and device. PyTorch registers
the module for device placement, optimization and checkpoint state.

```python
from src.models.cnn import ShallowEEGCNN

model = ShallowEEGCNN(**model_config, front_end=dsf_module)
# Pass model to src.training.cnn.fit with the shared training/masking settings.
# Restore with the same model_config and architecture for dsf_module.
model.load_state_dict(checkpoint["model_state_dict"])
# src.evaluation.pipeline.run(args, front_end=dsf_module) also loads these weights.
```
