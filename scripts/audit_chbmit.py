"""Channel-montage and annotation audit over every CHB-MIT record.

Reads only the EDF headers of the local recordings. Writes to results/audit/:
  files.csv      one row per EDF: labels, montage match, durations, seizure counts, flags
  seizures.csv   one row per annotated seizure (event_id, onset, offset, duration)
  patients.csv   per subject: usable files/hours/seizures under the 18-derivation montage
  report.md      human-readable summary of inconsistencies

  python scripts/audit_chbmit.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

import _common  # noqa: F401
from src.data.annotations import load_summary, summary_duration_s
from src.data.config import REPO_ROOT, load_config
from src.data.edf_io import read_edf_header
from src.data.montage import canonical, match_montage

DURATION_TOLERANCE_S = 2.0


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["dataset"]["raw_dir"])
    out = REPO_ROOT / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    derivations = cfg["montage"]["derivations"]
    fs_expected = float(cfg["dataset"]["sampling_rate_hz"])
    same_subject = cfg["dataset"].get("same_subject", {})

    records = (raw / "RECORDS").read_text().split()
    with_seizures = set((raw / "RECORDS-WITH-SEIZURES").read_text().split())
    summaries = {}
    for p in sorted({r.split("/")[0] for r in records}):
        summaries.update(load_summary(raw / p / f"{p}-summary.txt"))

    rows, seizure_rows = [], []
    for rel in records:
        patient, file_id = rel.split("/")[0], Path(rel).name.removesuffix(".edf")
        header = read_edf_header(raw / rel) if (raw / rel).exists() else None
        entry = summaries.get(file_id)
        flags = []
        row = {"record": rel, "patient_id": patient, "subject_id": same_subject.get(patient, patient),
               "file_id": file_id}

        if entry is None:
            flags.append("no_summary_entry")
        else:
            row["n_seizures"] = len(entry.seizures)
            row["seizure_s"] = sum(s.duration_s for s in entry.seizures)
            row["summary_duration_s"] = summary_duration_s(entry)
            if (len(entry.seizures) > 0) != (rel in with_seizures):
                flags.append("seizure_list_mismatch")
            for s in entry.seizures:
                seizure_rows.append({"event_id": s.event_id, "subject_id": row["subject_id"], "patient_id": patient,
                                     "file_id": file_id, "onset_s": s.onset_s, "offset_s": s.offset_s,
                                     "duration_s": s.duration_s})

        if header is None:
            flags.append("no_header")
        else:
            m = match_montage(header.labels, derivations)
            rates = sorted(set(header.sampling_rates))
            row.update({
                "n_signals": header.n_signals,
                "duration_s": header.duration_s,
                "sampling_rates": ";".join(f"{r:g}" for r in rates),
                "montage_complete": m.complete,
                "missing": ";".join(m.missing),
                "duplicates": ";".join(f"{k}x{len(v)}" for k, v in m.duplicates.items()),
                "n_dummy": m.n_dummy,
                "extras": ";".join(m.extras),
                "labels": "|".join(header.labels),
            })
            if not m.complete:
                flags.append("montage_incomplete")
            if rates != [fs_expected]:
                flags.append("sampling_rate")
            if entry is not None:
                if entry.channels and tuple(map(canonical, entry.channels)) != tuple(map(canonical, header.labels)):
                    flags.append("summary_channels_differ")
                if row["summary_duration_s"] is not None and abs(row["summary_duration_s"] - header.duration_s) > DURATION_TOLERANCE_S:
                    flags.append("summary_duration_differs")
                if any(s.offset_s > header.duration_s for s in entry.seizures):
                    flags.append("seizure_outside_recording")
        row["flags"] = ";".join(flags)
        row["usable"] = not (set(flags) & {"no_summary_entry", "no_header", "montage_incomplete", "sampling_rate",
                                           "seizure_outside_recording"})
        rows.append(row)

    files = pd.DataFrame(rows)
    seizures = pd.DataFrame(seizure_rows)
    files.to_csv(out / "files.csv", index=False)
    seizures.to_csv(out / "seizures.csv", index=False)

    use = files[files["usable"]]
    patients = files.groupby("subject_id").agg(
        case_folders=("patient_id", lambda s: ";".join(sorted(set(s)))),
        n_files=("record", "size"),
        n_usable=("usable", "sum"),
    ).join(use.groupby("subject_id").agg(
        usable_hours=("duration_s", lambda s: round(s.sum() / 3600, 2)),
        usable_seizures=("n_seizures", "sum"),
        usable_seizure_files=("n_seizures", lambda s: int((s > 0).sum())),
        usable_seizure_min=("seizure_s", lambda s: round(s.sum() / 60, 1)),
    )).fillna(0)
    all_sz = files.groupby("subject_id")["n_seizures"].sum().rename("all_seizures")
    patients = patients.join(all_sz).reset_index()
    patients.to_csv(out / "patients.csv", index=False)

    write_report(out / "report.md", files, patients, derivations)
    print((out / "report.md").read_text(encoding="utf-8"))


def write_report(path: Path, files: pd.DataFrame, patients: pd.DataFrame, derivations: list[str]) -> None:
    flag_counts = Counter(f for fl in files["flags"] for f in fl.split(";") if f)
    missing_counts = Counter(d for m in files["missing"].dropna() for d in m.split(";") if d)
    extras_counts = Counter(e for x in files["extras"].dropna() for e in x.split(";") if e)
    lines = [
        "# CHB-MIT channel and annotation audit", "",
        f"Required montage ({len(derivations)} bipolar derivations): {', '.join(derivations)}", "",
        f"- Records: {len(files)}",
        f"- Usable under the 18-derivation montage: {int(files['usable'].sum())} files, "
        f"{files.loc[files['usable'], 'duration_s'].sum() / 3600:.1f} h, "
        f"{int(files.loc[files['usable'], 'n_seizures'].sum())} of {int(files['n_seizures'].sum())} annotated seizures",
        "", "## Flags", "",
        "| flag | files |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in flag_counts.most_common()],
        "", "## Derivations missing (file counts)", "",
        "| derivation | files missing it |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in missing_counts.most_common()],
        "", "## Duplicated derivations", "",
        *[f"- `{d}`: {n} files" for d, n in Counter(files["duplicates"].dropna()).most_common() if d],
        "", "## Labels outside the montage (top 25, file counts)", "",
        ", ".join(f"`{k}` ({v})" for k, v in extras_counts.most_common(25)),
        "", "## Files that are not usable", "",
        "| record | flags | missing |", "|---|---|---|",
        *[f"| {r.record} | {r.flags} | {r.missing if isinstance(r.missing, str) else ''} |"
          for r in files[~files["usable"]].itertuples()],
        "", "## Per subject", "",
        patients.to_markdown(index=False) if _has_tabulate() else patients.to_string(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
