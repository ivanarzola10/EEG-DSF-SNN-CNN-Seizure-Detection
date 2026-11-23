"""Proposal requirements: raw samples, training-only statistics, and frozen splits."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.annotations import Seizure
from src.data.preparation import choose_clean_cohort
from src.data.windowing import seizure_overlap_s


def test_overlap_counts_all_seizure_activity_once():
    events = [Seizure("file", 0, 0, 1), Seizure("file", 1, 3, 4)]
    assert seizure_overlap_s(0, 4, events) == 2
    events.append(Seizure("file", 2, .5, 1.5))
    assert seizure_overlap_s(0, 4, events) == 2.5
    assert seizure_overlap_s(4, 8, events) == 0


def test_clean_selection_is_deterministic_and_stratified():
    rows = []
    for i in range(1, 24):
        for j in range(2):
            rows.append(dict(subject_id=f"chb{i:02}", patient_id=f"chb{i:02}",
                             quality_status="problem" if i > 19 else "usable",
                             n_seizures=i if j == 0 else 0))
    data = pd.DataFrame(rows)
    splits, subjects, strata = choose_clean_cohort(data, {})
    assert splits == choose_clean_cohort(data.sample(frac=1, random_state=3), {})[0]
    assert [len(splits[k]) for k in ("train", "val", "test")] == [12, 3, 3]
    assert sum(s["selected"] for s in subjects) == 18
    for rank in range(3):
        assignments = [s["split"] for s in strata if s["stratum"] == rank]
        assert assignments.count("train") == 4
        assert assignments.count("val") == assignments.count("test") == 1


def test_prepare_preserves_unfiltered_samples_and_uses_training_only(tmp_path, monkeypatch):
    from scripts import prepare_dataset
    patients = [f"chb{i:02}" for i in range(1, 19)]
    splits = dict(train=patients[:12], val=patients[12:15], test=patients[15:])
    cfg = dict(dataset=dict(raw_dir="raw", sampling_rate_hz=256, same_subject={}),
               montage=dict(derivations=[f"channel{i}" for i in range(18)]),
               preprocessing={"units": "uV", "filtering": "none"})
    sources = [prepare_dataset.Source(tmp_path / patient / f"{patient}_01.edf", patient, split,
                                      1280, (Seizure(f"{patient}_01", 0, 1, 3),))
               for split, cases in splits.items() for patient in cases]
    original = np.arange(18 * 1280, dtype=np.float32).reshape(18, 1280) + 1234
    def recording(path, *args, **kwargs):
        patient = path.parent.name
        # Deliberately different held-out means to catch normalization leakage.
        x = original + (1_000_000 if patient not in splits["train"] else 0)
        return SimpleNamespace(data=x, extra={"duplicates_identical": {"dup": True}})
    monkeypatch.setattr(prepare_dataset, "load_recording", recording)
    spec = prepare_dataset.dataset_spec(cfg, dict(splits=splits), sources, [], False)
    root, _ = prepare_dataset.ensure_prepared(tmp_path / "prepared", spec, sources)
    np.testing.assert_array_equal(np.load(root / "recordings/chb01_01.npy"), original)
    with np.load(root / "normalization.npz") as norm:
        np.testing.assert_allclose(norm["mean"][:, 0], original.mean(axis=1))
        np.testing.assert_allclose(norm["std"][:, 0], original.std(axis=1), rtol=1e-6)
        assert norm["sample_count"] == 12 * 1280
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["data_config"]["preprocessing"] == {"units": "uV", "filtering": "none"}
    assert manifest["status"] == "complete"
    metadata = pd.read_csv(root / "metadata.csv")
    assert set(metadata.start_sample) == {0, 256}
    assert (metadata.end_s - metadata.start_s == 4).all()
    assert (metadata.label == 1).all()
    assert prepare_dataset.ensure_prepared(root, spec, sources)[0] == root
    metadata.loc[0, "label"] = 0
    metadata.to_csv(root / "metadata.csv", index=False)
    replacement, _ = prepare_dataset.ensure_prepared(root, spec, sources)
    assert replacement != root
    assert pd.read_csv(root / "metadata.csv").label.iloc[0] == 0
    assert prepare_dataset.ensure_prepared(root, spec, sources)[0] == replacement
