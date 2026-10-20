"""Plot the 18 filtered derivations of one recording, with annotated seizures shaded.

  python scripts/plot_recording.py chb01_03                     # 60 s around the first seizure
  python scripts/plot_recording.py chb01_01 --start 0 --end 30  # any span, in seconds

Writes results/figures/<file_id>_<start>-<end>s.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _common  # noqa: F401
from src.data.annotations import load_summary
from src.data.config import REPO_ROOT, bandpass_from_config, load_config
from src.data.edf_io import load_recording, patient_id_from_file_id
from src.data.preprocessing import causal_bandpass

PAD_S = 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_id", help="e.g. chb01_03")
    ap.add_argument("--start", type=float, help="seconds from file start")
    ap.add_argument("--end", type=float, help="seconds from file start")
    args = ap.parse_args()

    cfg = load_config()
    patient = patient_id_from_file_id(args.file_id)
    folder = Path(cfg["dataset"]["raw_dir"]) / patient
    derivs = cfg["montage"]["derivations"]
    rec = load_recording(folder / f"{args.file_id}.edf", derivs, cfg["dataset"]["sampling_rate_hz"])
    seizures = load_summary(folder / f"{patient}-summary.txt")[args.file_id].seizures
    x = causal_bandpass(rec.data, bandpass_from_config(cfg))

    if args.start is None:
        t0, t1 = (seizures[0].onset_s - PAD_S, seizures[0].offset_s + PAD_S) if seizures else (0, 60)
    else:
        t0, t1 = args.start, args.end if args.end is not None else args.start + 60
    t0, t1 = max(t0, 0), min(t1, rec.duration_s)
    s0, s1 = int(t0 * rec.fs), int(t1 * rec.fs)
    t = np.arange(s0, s1) / rec.fs
    spacing = 4 * float(np.median(x.std(1)))

    fig, ax = plt.subplots(figsize=(13, 8))
    for i in range(len(derivs)):
        ax.plot(t, x[i, s0:s1] - i * spacing, lw=0.4, color="0.15")
    for sz in seizures:
        ax.axvspan(sz.onset_s, sz.offset_s, color="#d6604d", alpha=0.18, lw=0)
    ax.set_yticks([-i * spacing for i in range(len(derivs))], derivs, fontsize=7)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-(len(derivs) - 0.5) * spacing, spacing)
    ax.set_xlabel("time since file start (s)")
    ax.set_title(f"{args.file_id}: causal 1–40 Hz bandpass, {len(seizures)} annotated seizure(s) "
                 f"{[(s.onset_s, s.offset_s) for s in seizures]}; spacing {spacing:.0f} µV", fontsize=10)
    fig.tight_layout()

    out = REPO_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.file_id}_{t0:g}-{t1:g}s.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
