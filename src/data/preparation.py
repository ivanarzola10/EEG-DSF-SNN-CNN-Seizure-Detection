"""Shared preparation checks and stable continuous-recording statistics."""
from __future__ import annotations

import numpy as np


def validate_splits(splits, same_subject):
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits")
    seen = set()
    for name, patients in splits.items():
        if not patients or len(patients) != len(set(patients)):
            raise ValueError(f"Empty or duplicated patients in {name}")
        subjects = {same_subject.get(p, p) for p in patients}
        if seen & subjects:
            raise ValueError("Patient identity occurs in multiple splits")
        seen.update(subjects)


class ChannelMoments:
    """Merge per-recording moments; every continuous sample counts exactly once."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all() or not x.shape[1]:
            raise ValueError("Expected finite nonempty [channels, samples]")
        count = x.shape[1]
        mean = x.mean(axis=1, keepdims=True)
        m2 = ((x - mean) ** 2).sum(axis=1, keepdims=True)
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
        else:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += m2 + delta**2 * self.count * count / total
            self.mean += delta * count / total
            self.count = total

    def finish(self):
        if not self.count:
            raise ValueError("No training samples available")
        return self.mean.astype(np.float32), np.sqrt(
            np.maximum(self.m2 / self.count, 1e-8)).astype(np.float32)


def choose_clean_cohort(quality, same_subject, seed=42):
    """Freeze 18 subjects with no observed integrity flags, ranked by coverage.

    `quality` is a per-record table; missing downloads are separate from damaged
    or inconsistent recordings. Subject aliases always travel together.
    """
    import pandas as pd
    rows = []
    for subject, files in quality.groupby("subject_id", sort=True):
        good = files[files["quality_status"] == "usable"]
        events = int(good["n_seizures"].fillna(0).sum())
        problems = int((files["quality_status"] == "problem").sum())
        rows.append(dict(subject_id=subject, case_folders=sorted(set(files.patient_id)),
                         recordings=len(files), usable_recordings=len(good),
                         missing_recordings=int((files["quality_status"] == "missing").sum()),
                         problem_recordings=problems, seizures=events,
                         coverage=len(good) / len(files),
                         eligible=problems == 0 and events > 0))
    subjects = pd.DataFrame(rows)
    eligible = subjects[subjects.eligible].sort_values(
        ["coverage", "usable_recordings", "subject_id"], ascending=[False, False, True])
    if len(eligible) < 18:
        raise ValueError(f"Only {len(eligible)} patients have no observed quality problems and retained seizures; need 18")
    chosen = eligible.head(18)
    selected = set(chosen.subject_id)
    subjects["selected"] = subjects.subject_id.isin(selected)
    # Rank into three equal-sized seizure-count strata. Each contributes 4/1/1
    # subjects to train/validation/test. Identity is the unit of assignment.
    ordered = chosen.sort_values(["seizures", "subject_id"]).to_dict("records")
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    strata = []
    for stratum in range(3):
        group = ordered[stratum * 6:(stratum + 1) * 6]
        rng.shuffle(group)
        for i, subject in enumerate(group):
            split = "train" if i < 4 else "val" if i == 4 else "test"
            splits[split].extend(subject["case_folders"])
            strata.append(dict(subject_id=subject["subject_id"], seizures=subject["seizures"],
                               stratum=stratum, split=split))
    splits = {s: sorted(p) for s, p in splits.items()}
    validate_splits(splits, same_subject)
    return splits, subjects.to_dict("records"), strata
