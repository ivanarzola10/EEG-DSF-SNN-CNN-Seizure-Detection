"""Prepare the canonical continuous EEG cache and validate it against raw inputs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import mkdtemp
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from src.data.annotations import Seizure, load_summary
from src.data.config import load_config
from src.data.edf_io import load_recording, read_edf_header
from src.data.montage import match_montage
from src.data.preparation import ChannelMoments, validate_splits
from src.data.windowing import seizure_overlap_s

CACHE = Path("data/processed/prepared_proposal")
PROTOCOL = "proposal_unfiltered_v1"
WINDOW_SECONDS = 4
STRIDE_SECONDS = 1
POSITIVE_FRACTION = 0.5
WINDOW_FIELDS = ("patient_id", "subject_id", "file_id", "split", "start_sample",
                 "start_s", "end_s", "label", "continuous_path")
EVENT_FIELDS = ("event_id", "patient_id", "file_id", "onset_s", "offset_s")


@dataclass(frozen=True)
class Source:
    path: Path
    patient_id: str
    split: str
    samples: int
    seizures: tuple[Seizure, ...]
    offset_samples: int = 0

    @property
    def file_id(self):
        return self.path.stem


def resolve_inputs(raw_dir=None):
    """Validate the frozen inventory, including every eligible EDF header and summary."""
    cfg = load_config()
    cfg["preprocessing"].pop("bandpass", None)  # Exploratory plotting only.
    if cfg["preprocessing"] != {"units": "uV", "filtering": "none"}:
        raise ValueError("The canonical protocol requires unfiltered EEG in microvolts")
    dataset = cfg["dataset"]
    raw = Path(raw_dir or dataset["raw_dir"]).resolve()
    dataset["raw_dir"] = str(raw)
    fs = dataset["sampling_rate_hz"]
    aliases = dataset["same_subject"]
    channels = cfg["montage"]["derivations"]
    if fs != 256 or len(set(channels)) != 18 or len(channels) != 18:
        raise ValueError("Expected 18 unique ordered derivations at 256 Hz in configs/data.yaml")
    if aliases.get("chb21") != "chb01":
        raise ValueError("The CHB-MIT alias chb21 must map to chb01")

    cohort = json.loads(Path("configs/cohort.json").read_text())
    splits = cohort["splits"]
    validate_splits(splits, aliases)
    assignment = {case: split for split, cases in splits.items() for case in cases}
    subjects = {aliases.get(case, case): split for case, split in assignment.items()}
    strata = cohort["strata"]
    if (len(subjects) != 18 or len(strata) != 18
            or {row["subject_id"]: row["split"] for row in strata} != subjects):
        raise ValueError("Expected the frozen 18 distinct patient assignments in configs/cohort.json")
    for rank in range(3):
        group = [row["split"] for row in strata if row["stratum"] == rank]
        if sorted(group) != ["test", "train", "train", "train", "train", "val"]:
            raise ValueError(f"Expected 4/1/1 train/validation/test patients in stratum {rank}")

    inventory = pd.read_csv("configs/recording_inventory.csv", keep_default_na=False)
    if inventory.file_id.duplicated().any():
        raise ValueError("Duplicate recording IDs in the frozen inventory")
    sources, unavailable = [], []
    summaries = {}
    counts = dict.fromkeys(subjects, 0)
    for row in inventory.itertuples(index=False):
        patient, file_id = row.patient_id, row.file_id
        if patient not in assignment:
            continue
        if row.quality_status == "missing":
            unavailable.append(dict(file_id=file_id, reason=row.quality_reason))
            continue
        if row.quality_status != "usable":
            raise ValueError(f"{file_id}: unexpected inventory status {row.quality_status}")
        path = raw / patient / f"{file_id}.edf"
        summary = raw / patient / f"{patient}-summary.txt"
        try:
            header = read_edf_header(path)
            if path.stat().st_size != header.expected_file_bytes:
                raise ValueError("EDF file size differs from its header. Restore the complete EDF")
            montage = match_montage(header.labels, channels)
            if montage.missing:
                raise ValueError(f"EDF lacks canonical derivations {montage.missing}")
            if {header.sampling_rates[i] for i in montage.indices} != {fs}:
                raise ValueError(f"Canonical channels must be sampled at {fs} Hz")
            samples = round(header.duration_s * fs)
            if patient not in summaries:
                summaries[patient] = load_summary(summary)
            entry = summaries[patient][file_id]
            events = tuple(entry.seizures)
            if (entry.n_seizures_declared is None or len(events) != float(row.n_seizures)
                    or any(not 0 <= e.onset_s < e.offset_s <= samples / fs for e in events)):
                raise ValueError(f"Unexpected seizure count or invalid interval in {summary}")
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}: check the EDF and its annotations in {summary}: {exc}") from exc
        counts[aliases.get(patient, patient)] += len(events)
        sources.append(Source(path, patient, assignment[patient], samples, events))
    if (set(assignment) != {source.patient_id for source in sources}
            or counts != {row["subject_id"]: row["seizures"] for row in strata}):
        raise ValueError("Raw recordings do not match the frozen case folders and seizure counts")
    return cfg, cohort, sources, unavailable


def smoke_sources(sources, fs):
    """Take a short real EDF excerpt around the first seizure in each split."""
    selected = []
    for split in ("train", "val", "test"):
        source = next(s for s in sources if s.split == split and s.seizures)
        event = source.seizures[0]
        start = max(0, int(event.onset_s) - 15)
        stop = min(source.samples, round((event.offset_s + 15) * fs))
        events = tuple(replace(e, onset_s=max(e.onset_s, start) - start,
                               offset_s=min(e.offset_s, stop / fs) - start)
                       for e in source.seizures if e.offset_s > start and e.onset_s < stop / fs)
        selected.append(replace(source, samples=stop - start * fs, seizures=events,
                                offset_samples=start * fs))
    return selected


def window_rows(source, cfg):
    fs = cfg["dataset"]["sampling_rate_hz"]
    aliases = cfg["dataset"]["same_subject"]
    size = WINDOW_SECONDS * fs
    for start in range(0, source.samples - size + 1, STRIDE_SECONDS * fs):
        start_s, end_s = start / fs, (start + size) / fs
        yield dict(patient_id=source.patient_id, subject_id=aliases.get(source.patient_id, source.patient_id),
                   file_id=source.file_id, split=source.split, start_sample=start, start_s=start_s,
                   end_s=end_s, continuous_path=f"recordings/{source.file_id}.npy",
                   label=int(seizure_overlap_s(start_s, end_s, source.seizures)
                             >= WINDOW_SECONDS * POSITIVE_FRACTION))


def event_rows(sources):
    return [dict(event_id=event.event_id, patient_id=source.patient_id, file_id=source.file_id,
                 onset_s=event.onset_s, offset_s=event.offset_s)
            for source in sources for event in source.seizures]


def dataset_spec(cfg, cohort, sources, unavailable, smoke):
    splits = cohort["splits"]
    if smoke:
        splits = {source.split: [source.patient_id] for source in sources}
    return dict(protocol=PROTOCOL, data_config=cfg, cohort_selection=cohort,
                splits=splits, unavailable=unavailable, excluded=[],
                window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS,
                positive_fraction=POSITIVE_FRACTION)


def prepare(out, spec, sources):
    cfg = spec["data_config"]
    channels = cfg["montage"]["derivations"]
    fs = cfg["dataset"]["sampling_rate_hz"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "recordings").mkdir()
    moments = ChannelMoments()
    included, train_records = [], []
    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        for source in sources:
            try:
                rec = load_recording(source.path, channels, expected_fs=fs)
                x = rec.data[:, source.offset_samples:source.offset_samples + source.samples]
                if not all(rec.extra.get("duplicates_identical", {}).values()):
                    raise ValueError("Duplicate derivations contain different signals")
                if x.shape != (len(channels), source.samples) or not np.isfinite(x).all():
                    raise ValueError("EDF signal shape differs from the header or contains nonfinite samples")
                rows = list(window_rows(source, cfg))
                if not rows:
                    raise ValueError("EDF is too short for a complete window")
            except (OSError, ValueError) as exc:
                raise ValueError(f"{source.path}: {exc}") from exc
            np.save(out / rows[0]["continuous_path"], x, allow_pickle=False)
            writer.writerows(rows)
            included.append(dict(file_id=source.file_id, patient_id=source.patient_id, split=source.split,
                                 samples=source.samples, windows=len(rows),
                                 positive_windows=sum(row["label"] for row in rows)))
            if source.split == "train":
                moments.update(x)
                train_records.append(source.file_id)
            print(f"Prepared {source.file_id}: {len(rows)} windows", flush=True)
    mean, std = moments.finish()
    np.savez(out / "normalization.npz", mean=mean, std=std,
             train_patients=spec["splits"]["train"], records=train_records,
             channels=channels, sample_count=moments.count,
             source="continuous_unfiltered_recordings", protocol=PROTOCOL)
    pd.DataFrame(event_rows(sources), columns=EVENT_FIELDS).to_csv(out / "seizures.csv", index=False)
    manifest = dict(**spec, status="complete", included=included,
                    normalization_records=train_records, split_counts=split_counts(included))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def split_counts(included):
    counts = {}
    for split in ("train", "val", "test"):
        rows = [r for r in included if r["split"] == split]
        counts[split] = dict(recordings=len(rows), windows=sum(r["windows"] for r in rows),
                             positive_windows=sum(r["positive_windows"] for r in rows))
    return counts


def validate_cache(root, spec, sources):
    """Check the complete cache schema and window labels against raw annotations."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Cache build did not complete")
    for key, value in spec.items():
        if manifest.get(key) != value:
            raise ValueError(f"Cache differs from the requested {key}")
    cfg = spec["data_config"]
    channels = cfg["montage"]["derivations"]
    pd.testing.assert_frame_equal(
        pd.read_csv(root / "seizures.csv"),
        pd.DataFrame(event_rows(sources), columns=EVENT_FIELDS), check_dtype=False, check_exact=True)
    metadata = pd.read_csv(root / "metadata.csv")
    if set(metadata.columns) != set(WINDOW_FIELDS):
        raise ValueError("Cache window metadata schema differs")
    groups = dict(tuple(metadata.groupby("file_id", sort=False)))
    if set(groups) != {source.file_id for source in sources}:
        raise ValueError("Cached recordings differ from the frozen inventory")

    included = []
    for source in sources:
        expected = pd.DataFrame(window_rows(source, cfg), columns=WINDOW_FIELDS)
        actual = groups[source.file_id].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[list(WINDOW_FIELDS)], expected, check_dtype=False, check_exact=True,
                                      obj=f"{source.file_id} windows")
        path = root / f"recordings/{source.file_id}.npy"
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if x.shape != (len(channels), source.samples) or x.dtype != np.float32:
            raise ValueError(f"{path}: invalid cache signal shape or dtype")
        included.append(dict(file_id=source.file_id, patient_id=source.patient_id, split=source.split,
                             samples=source.samples, windows=len(expected),
                             positive_windows=int(expected.label.sum())))
    if manifest["included"] != included or manifest["split_counts"] != split_counts(included):
        raise ValueError("Cache manifest recording or split counts differ")
    training = [source for source in sources if source.split == "train"]
    train_ids = [source.file_id for source in training]
    if manifest["normalization_records"] != train_ids:
        raise ValueError("Normalization manifest differs from the training inventory")
    with np.load(root / "normalization.npz", allow_pickle=False) as norm:
        if (norm["records"].tolist() != train_ids or norm["channels"].tolist() != channels
                or norm["train_patients"].tolist() != spec["splits"]["train"]
                or norm["sample_count"].item() != sum(s.samples for s in training)
                or norm["source"].item() != "continuous_unfiltered_recordings"
                or norm["protocol"].item() != spec["protocol"]
                or any(norm[k].shape != (len(channels), 1) or not np.isfinite(norm[k]).all()
                       for k in ("mean", "std")) or (norm["std"] <= 0).any()):
            raise ValueError("Cache normalization does not match the training recordings and montage")
    return manifest


def ensure_prepared(cache, spec, sources):
    cache.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (cache, *sorted(cache.parent.glob(f"{cache.name}-*"))):
        if not candidate.is_dir():
            continue
        try:
            return candidate, validate_cache(candidate, spec, sources)
        except (OSError, EOFError, BadZipFile, ValueError, KeyError, TypeError, AssertionError) as exc:
            print(f"Preserving incompatible cache {candidate}: {exc}", flush=True)
    out = Path(mkdtemp(prefix=f"{cache.name}-", dir=cache.parent)) if cache.exists() else cache
    if out == cache:
        out.mkdir(exist_ok=False)
    prepare(out, spec, sources)
    return out, validate_cache(out, spec, sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path, default=CACHE)
    args = parser.parse_args()
    cfg, cohort, sources, unavailable = resolve_inputs(args.raw_dir)
    path, manifest = ensure_prepared(args.output, dataset_spec(cfg, cohort, sources, unavailable, False), sources)
    print(path, json.dumps(manifest["split_counts"], indent=2))


if __name__ == "__main__":
    main()
