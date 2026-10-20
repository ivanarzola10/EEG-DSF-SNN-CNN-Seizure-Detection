# EEG Seizure Detection Under Missing Channels — Project Handoff

**Status:** GitHub repository created; implementation not yet started.  
**Team:** Two people.  
**Scope:** Course project first, with potential follow-up research paper.  
**Updated:** October 19, 2026.

> **Instructions to the next coding/research agent:** Continue this project incrementally in the user's repository. Start with dataset ingestion and verification, not model training. Keep the pipeline causal, patient-disjoint, reproducible, and small enough for a two-person course project. Distinguish established paper findings from our proposed design choices. Do not claim novelty or energy savings without evidence.

## 1. Project objective

Study whether **Dynamic Spatial Filtering (DSF)** makes EEG seizure detection more robust when input channels are missing, and whether DSF provides a different benefit for **Spiking Neural Networks (SNNs)** than for conventional **CNNs**.

**Primary research question:**

> Does DSF improve seizure detection under simulated EEG channel loss on CHB-MIT, and is the improvement different for CNN and SNN models?

Key requirements:

- **Unseen-patient evaluation:** split by patient **before** windowing or fitting any preprocessing parameters.
- **Causal, low-latency inference:** use current/past samples only; report predictions at window end.
- **Fair comparisons:** same data, labels, masking distributions, test masks, and evaluation pipeline for all models.
- **Reliable event detection:** evaluate seizure events and false alarms, not accuracy alone.

This work tests **seizure detection** (recognizing an ongoing seizure), **not prediction** before onset.

## 2. Dataset

**Source:** CHB-MIT Scalp EEG Database (PhysioNet): https://physionet.org/content/chbmit/1.0.0/

- EDF EEG recordings with separate seizure annotation summaries (`chbXX-summary.txt`).
- Most recordings contain ~23 EEG signals at **256 Hz**, but channel sets vary.
- **Proposed common montage:** 18 bipolar derivations, *conditional on verifying availability and naming across every selected file*:

```text
FP1-F7   F7-T7   T7-P7   P7-O1
FP1-F3   F3-C3   C3-P3   P3-O1
FP2-F8   F8-T8   T8-P8   P8-O2
FP2-F4   F4-C4   C4-P4   P4-O2
FZ-CZ    CZ-PZ
```

These are **18 bipolar derivations, not 18 physical electrodes**. Some recordings have channel-name/montage differences. Do not silently force a standard montage: produce an audit first and either choose recordings with consistent labels or document a deterministic harmonization policy. `chb01` and `chb21` represent the same subject and **must remain in the same split**.

**Start small:** Download one non-seizure EDF (`chb01_01.edf`), one seizure EDF (`chb01_03.edf`), and `chb01-summary.txt`. After validating the loader, audit a manageable multi-patient subset with seizure and seizure-free recordings. A ~12-patient pilot is an initial *suggestion*, not a fixed decision; ensure train, validation, and test each contain sufficient annotated events. Preserve full recordings for event-level evaluation.

## 3. Common preprocessing and splits

| Parameter | Proposed starting choice |
|---|---|
| Sampling rate | 256 Hz (native) |
| Input channels | Fixed verified set of ~18 bipolar derivations |
| Filtering | Causal bandpass, e.g. 1–40 Hz; no zero-phase/future-aware filtering |
| Window length | 4 seconds = 1,024 samples at 256 Hz |
| Window stride | 1 second |
| Normalization | Per-channel statistics estimated from training patients only |
| Input shape | `[channels, time]`, e.g. `[18, 1024]` |
| Labels | Binary seizure / non-seizure, derived from annotated onset/offset |

**Critical implementation rules:**

1. Audit EDF channel labels, duplicated derivations, seizure timestamps, file lengths, and patient identity.
2. Freeze train/validation/test patient assignments **before** extracting overlapping windows.
3. Fit normalization on training data only; reuse fixed training statistics everywhere.
4. Define and unit-test an explicit window-label rule for partial seizure overlap. Store window start/end timestamps and event IDs; do not label with information that would not yet be observable at prediction time.
5. Run causal filtering with correctly handled state along the continuous recording; do not introduce future samples or inconsistent window-edge artifacts.
6. Keep validation/test recordings at their natural seizure prevalence; do not oversample test data.
7. Store random seeds, patient splits, channel list, annotations, filter settings, and normalization statistics in versioned configs/manifests.

## 4. Core models and controlled experiment

Train **all four** models using the **same channel-masking augmentation**:

| ID | Model | Spatial adaptation |
|---|---|---|
| A | Compact causal CNN | None |
| B | Compact causal CNN + DSF | DSF before CNN |
| C | Convolutional SNN | None |
| D | Convolutional SNN + DSF | DSF before SNN's input projection/spike encoding |

### CNN

Start with a compact causal 1D CNN (2–3 convolutional layers and classifier). Train first with intact channels to verify the full pipeline; then train the **masking-trained** version for the controlled study. Address class imbalance in training (e.g., weighted loss or training sampler), not by changing test prevalence.

### SNN

Use PyTorch + `snnTorch` (or another clearly documented SNN framework) with a compact temporal convolutional architecture and LIF neurons, trained with surrogate gradients. A causal temporal projection/downsampling stage is acceptable if needed to keep the number of simulated steps manageable. Record the chosen spike encoder, timesteps, parameter count, and preprocessing. Match the CNN's temporal information and broadly comparable model capacity where feasible.

### DSF

Start from Banville et al. (2022), *Robust Learning from Corrupted EEG with Dynamic Spatial Filtering*: https://doi.org/10.1016/j.neuroimage.2022.118994

Code reference: https://github.com/hubertjb/dynamic-spatial-filtering

- Start with **log-variance-based DSF** as the simpler configuration; covariance-based DSF is optional.
- DSF processes **continuous-valued EEG** before the SNN spike-generation stage, so that pipeline is **hybrid**.
- Handle exactly zeroed channels safely (`eps` for logarithms; explicit availability mask if using it).
- If one model receives a channel-availability indicator, document this and ensure fair comparisons with other models.
- Count DSF overhead in any inference cost or energy discussion.

**Optional comparator (after the four core models work):** A lightweight **mask-aware channel-attention layer**. It is a useful test of whether DSF outperforms a simpler spatial mechanism; do not delay the core project to implement it. Other later comparators include a static learned spatial layer and channel-independent masked aggregation.

## 5. Simulating missing channels

Two **distinct** robustness experiments are planned.

### A. Random missing *bipolar channels* (core experiment)

- Start with normalized input `X` of shape `[18, 1024]`.
- Select a subset of channel indices and replace those channels with **0.0** for the full window/recording.
- Return an explicit binary **availability mask** (`1` available, `0` unavailable).
- Test **0%, approximately 25%, 50%, and 75%** missing channels.
  - With 18 channels and `ceil(fraction * 18)`: 0, 5, 9, and 14 removed, respectively.
- During **training**, sample randomized masks and include some clean examples, using **the same policy for every model**.
- During **testing**, use pre-generated, reproducible masks. Apply each mask to an **entire EDF recording** (not a newly randomized subset at every 1-second window) and repeat over multiple subsets. Evaluate all models on the **identical masks**.

Example masking logic:

```python
import numpy as np

def mask_channels(x, missing_fraction, rng):
    """x: normalized EEG [C, T]; returns masked_x, available_mask."""
    c = x.shape[0]
    k = int(np.ceil(c * missing_fraction))
    missing = rng.choice(c, size=k, replace=False)
    available = np.ones(c, dtype=np.float32)
    available[missing] = 0.0
    masked = x.copy()
    masked[missing, :] = 0.0
    return masked, available
```

### B. Simulated *physical electrode failure* (extension)

A physical electrode is involved in more than one bipolar derivation. For example, losing **F7** invalidates **FP1-F7** and **F7-T7** in the proposed montage. Build an electrode-to-derivation map and zero **all affected bipolar channels together**. Later try intermittent disconnections and noisy-but-present signals. This is still simulated failure, **not validation on naturally corrupted wearable recordings**.

## 6. Evaluation and fairness

**Primary outcomes:**

- **Window-level PR-AUC** on held-out patients.
- **Seizure-event sensitivity** (fraction of annotated seizure events detected).
- **False alarms per hour** during non-seizure recording time.
- **Detection delay** from seizure onset to first qualifying alert.
- **End-to-end inference time**, model size, and computational overhead; SNN spike activity can be reported separately.

Choose probability thresholds and alert-smoothing/cooldown rules on **validation data only**, then freeze all rules for testing. Align predicted time with the **end of each 4-second window**. Evaluate continuous recordings in chronological order. Report patient-level results and variability across random mask subsets; overlapping windows are **not independent patients**.

Primary comparisons:

- `CNN+DSF` vs `CNN`, under the same mask augmentation.
- `SNN+DSF` vs `SNN`, under the same mask augmentation.
- Change in DSF benefit for the SNN relative to the CNN.
- Reliability and computation trade-offs at increasing missing-channel rates.

**Do not claim** a model is more energy-efficient based only on fewer spikes. End-to-end computation includes preprocessing, DSF, spike encoding, and readout; hardware energy would require separate measurement/appropriate validated estimates.

## 7. Suggested milestones and ownership

| Week | Main work | Completion criterion |
|---|---|---|
| 1 | Dataset audit, loader, seizure annotations, splits, causal preprocessing | Verified plots and labeled windows from seizure and non-seizure EDFs |
| 2 | Compact CNN; optional Welch bandpower + logistic regression baseline | Reproducible CNN validation results |
| 3 | Compact convolutional SNN | SNN trains and runs through the same evaluator |
| 4 | Shared masking policy and DSF integration | All four masking-trained model variants work |
| 5 | Fixed-mask robustness experiment; physical electrode-loss extension | Comparable PR-AUC, event metrics, and latency tables/plots |
| 6 | Ablations, patient-level analysis, report | Reproducible experiment results and final course report |

**Suggested division:** Person A owns dataset parsing/preprocessing, missing-channel simulator, and event evaluation. Person B owns CNN/SNN training, DSF integration, and inference profiling. Both review leakage/causality issues and experiment results.

## 8. Proposed repository structure

```text
.
├── PROJECT_HANDOFF.md
├── README.md
├── requirements.txt                 # or pyproject.toml
├── .gitignore                       # ignore data, checkpoints, large results
├── configs/
│   ├── data.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/                         # local only, not committed
│   └── processed/                   # local only, not committed
├── src/
│   ├── data/                        # EDF IO, annotations, splits, preprocessing
│   ├── models/                      # CNN, SNN, DSF, optional attention
│   ├── training/                    # reproducible training functions
│   └── evaluation/                  # window/event metrics, alarms, plots
├── scripts/
│   ├── audit_chbmit.py
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── tests/                            # channel maps, label bounds, masks, leakage
└── results/                          # concise tracked summaries; large outputs ignored
```

The filenames above are **proposed**. Adapt to the repo's existing structure rather than overwriting user work.

## 9. Immediate tasks for the next agent — START HERE

**Do not start by implementing all four networks. First make the data trustworthy.**

1. Inspect the repository and existing files; preserve current work. Check `.gitignore` so full EDF data and model checkpoints are **not committed**.
2. Set up a reproducible Python environment with `mne`, `numpy`, `scipy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`, `snntorch`, and `pytest` (install dependencies only as needed).
3. Obtain the small CHB-MIT sample: `chb01_01.edf`, `chb01_03.edf`, `chb01-summary.txt`. Confirm download URLs against PhysioNet; avoid pulling the entire dataset initially.
4. Implement EDF inspection: print recording duration, sampling rate, exact channel names, and EEG signal shape.
5. Parse summary annotations into seizure intervals and plot seizure vs. seizure-free segments. Check against source timestamps.
6. Implement a **channel-montage audit** over available files. Report which recordings contain the proposed 18 derivations and any naming inconsistencies.
7. Implement causal preprocessing and 4-second windows with a 1-second stride. Return the signal, label, patient ID, file ID, start time, and end time. Unit-test labels and boundary conditions.
8. Propose a **patient-disjoint split** only after reviewing the multi-patient seizure/recording audit. Save its manifest; do not use random window splitting.
9. Show the user: (a) seizure plot, (b) channel audit, (c) window counts by class and patient, (d) exact scripts to reproduce. **Then** move to the compact CNN.

**Definition of first success:** A deterministic pipeline that loads CHB-MIT, identifies an annotated seizure, prints verified channel information, and produces correctly timed `(channels, 1024)` labeled windows without leakage.

## 10. References and novelty constraints

Papers previously discussed:

1. **Banville et al. (2022)** — Dynamic spatial filtering under EEG corruption, *NeuroImage*. https://doi.org/10.1016/j.neuroimage.2022.118994
2. **EESNN (2024)** — Cross-patient seizure detection with a spiking network. https://pmc.ncbi.nlm.nih.gov/articles/PMC10805904/
3. **Attention-based CNN-BiRNN (BIBM 2019)** — EEG channel dropout/missing-channel experiments. https://ieeexplore.ieee.org/document/8983420
4. **Event-driven neuromorphic seizure system (2025)** — SNN with neuromorphic processing, intracranial EEG. https://www.nature.com/articles/s41598-025-99272-6
5. **EEG gap-imputation review (Sensors 2026)** — Focuses on missing temporal segments, not ordinary full-channel dropout. https://www.mdpi.com/1424-8220/26/8/2431
6. **HR-SNN (2024)** — SNN motor-imagery study with channel-loss/noise experiments; not seizure detection. https://doi.org/10.1109/TCDS.2024.3395443
7. **BIOT (NeurIPS 2023)** — Channel-varying biosignal representation learning, includes CHB-MIT. https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html

**Claim carefully:** Channel masking is not novel. DSF has been evaluated with CNNs for other EEG tasks. Existing SNNs have been evaluated under missing-channel conditions in other EEG domains. We have **not established** that this is the first DSF + SNN seizure detector; novelty requires another comprehensive literature check. The plausible contribution is a well-controlled comparison of robust seizure detection on unseen patients, especially missing bipolar derivations versus simulated electrode failures and their accuracy–latency trade-offs.

## 11. Out of scope for the first course iteration

- Seizure forecasting/prediction before onset.
- JEPA or large self-supervised pretraining systems.
- Clinical validation or claims about real wearable failure rates.
- Neuromorphic hardware deployment or unmeasured energy claims.
- Large graph networks, generative channel imputation, or extensive architecture searches before baseline results.

**Decision rule:** Complete the core CNN/SNN × with/without DSF comparison first. Add mask-aware attention and electrode-failure simulations only after the end-to-end pipeline is reliable.
