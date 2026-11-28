"""CNN calibration and held-out evaluation with frozen masks and alarm rules."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.channel_masks import make_mask_manifest, validate_mask_manifest, RecordingMaskedDataset, validate_experiment
from src.data.preparation import validate_splits
from src.data.torch_dataset import WindowDataset
from src.evaluation.energy import SpikeDensityProbe, dense_macs, energy_estimate, weighted_firing_rate
from src.evaluation.events import calibrate_rules, recording_events, summarize_events, validate_alarm_policy
from src.models.dsf import DynamicSpatialFilter, ImportanceProbe
from src.models.registry import build_model
from src.training.cnn import binary_metrics, choose_device, evaluate


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prediction_recordings(dataset, probabilities, manifest, annotations):
    """Group chronologically, retaining every original seizure and true duration."""
    if len(probabilities) != len(dataset):
        raise ValueError("Prediction count differs from dataset")
    durations = {r["file_id"]: r["samples"] / 256 for r in manifest["included"]}
    events = {file_id: group[["event_id", "onset_s", "offset_s"]].to_dict("records")
              for file_id, group in annotations.groupby("file_id")}
    records = []
    aliases = manifest["data_config"]["dataset"]["same_subject"]
    for file_id, rows in dataset.df.groupby("file_id", sort=True):
        rows = rows.sort_values("end_s")
        patient = rows.iloc[0].patient_id
        records.append(dict(file_id=file_id, patient_id=patient, subject_id=aliases.get(patient, patient),
                            end_times=rows.end_s.to_numpy(), probabilities=probabilities[rows.index.to_numpy()],
                            seizures=events.get(file_id, []), duration_seconds=durations[file_id]))
    return records


def score_recording(record, rule):
    return recording_events(**{k: record[k] for k in ("end_times", "probabilities", "seizures", "duration_seconds")}, rule=rule)


def patient_results(dataset, labels, probabilities, records, rule, fraction, repeat):
    aliases = {r["patient_id"]: r["subject_id"] for r in records}
    subjects = dataset.df.patient_id.map(aliases).to_numpy()
    patients, event_rows, alert_rows = [], [], []
    for subject in sorted(set(subjects)):
        matching = [r for r in records if r["subject_id"] == subject]
        scored = [score_recording(r, rule) for r in matching]
        identity = dict(subject_id=subject, missing_fraction=fraction, repeat=repeat)
        mask = subjects == subject
        patients.append({**identity, **binary_metrics(labels[mask], probabilities[mask]), **summarize_events(scored)})
        for record, result in zip(matching, scored):
            event_rows.extend({**identity, "file_id": record["file_id"], **event} for event in result["events"])
            alert_rows.extend({**identity, "file_id": record["file_id"], **alert} for alert in result["alerts"])
    return patients, event_rows, alert_rows


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(model, dataset, device, batch_size, iterations=10):
    x = torch.stack([dataset[i % len(dataset)][0] for i in range(batch_size)]).to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(x)
        synchronize(device)
        elapsed = time.perf_counter() - start
    accelerator = (torch.cuda.get_device_name(device) if device.type == "cuda"
                   else "Apple Metal" if device.type == "mps" else None)
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        import subprocess
        hardware = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        cpu = hardware.stdout.strip() or cpu
    front_end = getattr(model, "front_end", None)
    front_params = sum(p.numel() for p in front_end.parameters()) if front_end is not None else 0
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                front_end_parameters=front_params,  # DSF does not spike: reported separately
                backbone_parameters=sum(p.numel() for p in model.parameters()) - front_params,
                batch_size=batch_size,
                input_condition="complete_channels",
                warmup_batches=3, measured_batches=iterations, seconds_per_window=elapsed / (iterations * batch_size),
                scope="model forward only; includes any attached DSF; excludes disk IO, normalization, masking and alarm processing",
                device=str(device), cpu=cpu, accelerator=accelerator,
                architecture=platform.machine(), platform=platform.platform(),
                torch_threads=torch.get_num_threads(), torch_version=str(torch.__version__))


def summarize_patients(patient_rows):
    frame = pd.DataFrame(patient_rows)
    metrics = ["pr_auc", "average_precision", "roc_auc", "sensitivity", "specificity", "gmean", "raccuracy",
               "event_sensitivity", "false_alarms_per_hour",
               "mean_detection_delay_s", "median_detection_delay_s", "missed_seizures"]
    baseline = frame[frame.missing_fraction == 0].set_index("subject_id")
    for metric in metrics:
        frame[f"{metric}_change_from_complete"] = (
            pd.to_numeric(frame[metric]) - pd.to_numeric(frame.subject_id.map(baseline[metric]))
        )
    metrics += [f"{metric}_change_from_complete" for metric in metrics]
    summary = {}
    for fraction, group in frame.groupby("missing_fraction"):
        summary[str(fraction)] = dict(patients=int(group.subject_id.nunique()), mask_repeats=int(group.repeat.nunique()),
                                      patient_subset_rows=len(group), metrics={})
        for metric in metrics:
            values = group[metric].dropna().astype(float).to_numpy()
            summary[str(fraction)]["metrics"][metric] = dict(
                mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=0)) if len(values) else None,
                defined_rows=len(values))
    return frame, summary


def run(args, *, front_end=None):
    """Evaluate a checkpoint using the same optional module used to train it."""
    root = args.data_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest.get("protocol") != "proposal_unfiltered_v1":
        raise ValueError("Use a completed proposal dataset")
    validate_splits(manifest["splits"], manifest["data_config"]["dataset"]["same_subject"])
    experiment = json.loads(args.experiment_config.read_text())
    validate_experiment(experiment)
    policy = experiment["alarm_calibration"]
    validate_alarm_policy(policy)
    data_hashes = {name: fingerprint(root / name) for name in ("manifest.json", "metadata.csv", "normalization.npz", "seizures.csv")}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for name, expected in checkpoint["provenance"]["fingerprints"].items():
        if name in data_hashes and data_hashes[name] != expected:
            raise ValueError(f"Checkpoint was trained on different dataset artifact: {name}")
    if checkpoint["config"].get("channel_masking", "none") == "mixed" and checkpoint["config"]["experiment"] != experiment:
        raise ValueError("Experiment policy differs from checkpoint training policy")
    checkpoint_hash = fingerprint(args.checkpoint)
    args.mask_manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.mask_manifest.exists():
        masks = json.loads(args.mask_manifest.read_text())
        validate_mask_manifest(masks, manifest, experiment, data_hashes["manifest.json"])
    else:
        masks = make_mask_manifest(manifest, experiment, data_hashes["manifest.json"])
        write_json(args.mask_manifest, masks)
    mask_hash = fingerprint(args.mask_manifest)
    calibration = None
    if args.stage == "test":
        if args.calibration is None:
            raise ValueError("Test evaluation requires a frozen --calibration file")
        calibration = json.loads(args.calibration.read_text())
        if (calibration["checkpoint_sha256"] != checkpoint_hash or calibration["data_fingerprints"] != data_hashes
                or calibration["mask_manifest_sha256"] != mask_hash or calibration["experiment"] != experiment):
            raise ValueError("Calibration does not match the checkpoint, dataset, masks and experiment")
    split = "val" if args.stage == "calibrate" else "test"
    dataset = WindowDataset(root / "metadata.csv", manifest["splits"][split], root / "normalization.npz")
    if set(dataset.df.split) != {split} or len(dataset) != manifest["split_counts"][split]["windows"]:
        raise ValueError("Evaluation split does not match manifest")
    annotations = pd.read_csv(root / "seizures.csv")
    annotations = annotations[annotations.file_id.isin(dataset.df.file_id)]
    device = choose_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    model = build_model(checkpoint["config"], front_end=front_end).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    spiking = hasattr(model, "reset_spike_stats")
    dense = dense_macs(model, model.input_shape)
    probe = SpikeDensityProbe(model) if spiking else None
    dsf = getattr(model, "front_end", None)
    dsf_probe = ImportanceProbe(dsf) if isinstance(dsf, DynamicSpatialFilter) else None
    channel_names = masks["channels"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    provenance = dict(status="running", stage=args.stage, checkpoint_sha256=checkpoint_hash,
                      data_fingerprints=data_hashes, mask_manifest_sha256=mask_hash, experiment=experiment,
                      training_channel_masking=checkpoint["config"].get("channel_masking", "none"),
                      model_type=checkpoint["config"].get("model_type", "cnn"),
                      dsf=checkpoint["config"].get("dsf"),
                      timestamp_utc=datetime.now(timezone.utc).isoformat(), split=split)
    write_json(output / "provenance.json", provenance)
    write_json(output / "mask_manifest.json", masks)
    conditions = [(0.0, 0)] if args.stage == "calibrate" else [
        (f, r) for f in experiment["missing_fractions"]
        for r in range(1 if f == 0 else experiment["evaluation_repeats"])]
    patients, event_rows, alert_rows, pooled, energies, importance_rows = [], [], [], [], [], []
    for fraction, repeat in conditions:
        print(f"{split}: missing={fraction:.0%}, subset={repeat}; windows={len(dataset)}", flush=True)
        masked = RecordingMaskedDataset(dataset, masks, split, fraction, repeat)
        loader = DataLoader(masked, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        if spiking:
            model.reset_spike_stats()
            probe.reset()
        if dsf_probe is not None:
            dsf_probe.reset()
        metrics, labels, probabilities = evaluate(model, loader, device, args.log_every)
        # DSF only: normalized effective channel importance (Banville et al. Eq. 5), averaged
        # over windows, for available vs masked channels -- does DSF down-weight lost channels?
        dsf_columns = {}
        if dsf_probe is not None:
            phi = dsf_probe.summary()
            dsf_columns = dict(dsf_importance_available=phi["available_mean"],
                               dsf_importance_missing=phi["missing_mean"])
            importance_rows.extend(dict(missing_fraction=fraction, repeat=repeat, channel=name,
                                        importance_available=phi["available"][i],
                                        importance_missing=phi["missing"][i])
                                   for i, name in enumerate(channel_names))
        # SNN only: mean firing rate per hidden layer over this condition's windows, and the
        # neuron-weighted rate over all hidden neurons (the paper's "average spiking probability").
        spikes = {}
        if spiking:
            rates = model.spike_stats()["eval"]
            spikes = {f"firing_rate_layer{i + 1}": rate for i, rate in enumerate(rates)}
            spikes["firing_rate_mean"] = weighted_firing_rate(rates, model.neurons_per_layer())
        energy = energy_estimate(dense, probe.densities() if spiking else None)
        energy["front_end_macs"] = dense.get("front_end", 0.0)  # DSF MACs (eigendecomposition not counted)
        energies.append(dict(missing_fraction=fraction, repeat=repeat, **energy, **spikes))
        operations = {k: energy[k] for k in ("macs", "accumulates", "synaptic_operations", "energy_mj", "dense_energy_mj")}
        np.savez_compressed(output / "predictions" / f"missing_{fraction:.2f}_repeat_{repeat:02}.npz",
                            labels=labels, probabilities=probabilities, window_row=np.arange(len(dataset)))
        records = prediction_recordings(dataset, probabilities, manifest, annotations)
        if args.stage == "calibrate":
            inputs = [{k: r[k] for k in ("end_times", "probabilities", "seizures", "duration_seconds")} for r in records]
            selected, candidates = calibrate_rules(inputs, policy)
            calibration = dict(checkpoint_sha256=checkpoint_hash, data_fingerprints=data_hashes,
                               mask_manifest_sha256=mask_hash, experiment=experiment, split="val",
                               rule=selected["rule"], validation_event_metrics=selected["metrics"],
                               validation_window_metrics=metrics,
                               selection="maximize pooled validation event sensitivity under FA/h target; ties: lower mean delay, fewer FA/h, higher threshold",
                               test_set_evaluated=False)
            write_json(output / "calibration.json", calibration)
            write_json(output / "calibration_candidates.json", candidates)
            print(f"Selected alarm rule: {selected['rule']}; validation event metrics: {selected['metrics']}", flush=True)
        rows, events, alerts = patient_results(dataset, labels, probabilities, records, calibration["rule"], fraction, repeat)
        pooled.append(dict(missing_fraction=fraction, repeat=repeat, **metrics, **spikes, **operations, **dsf_columns,
                           **summarize_events([score_recording(r, calibration["rule"]) for r in records])))
        patients.extend(rows)
        event_rows.extend(events)
        alert_rows.extend(alerts)
    if probe is not None:
        probe.remove()  # keep the timing benchmark free of hook overhead
    if dsf_probe is not None:
        dsf_probe.remove()
        pd.DataFrame(importance_rows).to_csv(output / "dsf_channel_importance.csv", index=False)
    frame, summary = summarize_patients(patients)
    pooled_frame = pd.DataFrame(pooled)
    pr_auc = pd.to_numeric(pooled_frame.pr_auc)
    pooled_frame["pr_auc_change_from_complete"] = pr_auc - pr_auc.iloc[0]
    pooled_frame.to_csv(output / "pooled_metrics.csv", index=False)
    frame.to_csv(output / "patient_metrics.csv", index=False)
    pd.DataFrame(event_rows, columns=["subject_id", "missing_fraction", "repeat", "file_id", "event_id", "onset_s", "offset_s", "detected", "detected_at_s", "delay_s"]).to_csv(output / "seizure_events.csv", index=False)
    pd.DataFrame(alert_rows, columns=["subject_id", "missing_fraction", "repeat", "file_id", "time_s", "false_alarm"]).to_csv(output / "alerts.csv", index=False)
    write_json(output / "summary.json", dict(conditions=summary,
               aggregation="unweighted mean/std over patient and channel-subset rows; undefined values omitted; std is descriptive, not a confidence interval",
               delay_scope="only detected seizures; missed seizures reported separately in event and patient tables",
               alarm_semantics="scores strictly above threshold; consecutive 1-second windows; one alert per above-threshold run with causal cooldown; state resets at each EDF; event matching is onset-inclusive/offset-exclusive",
               false_alarm_denominator="entire included EDF duration minus annotated seizure duration, including startup time",
               timing=benchmark(model, dataset, device, args.batch_size),
               energy=energies[0],  # complete channels; every condition is in pooled_metrics.csv
               paper_metrics=("window-level at fixed p > 0.5 (Zhang et al. 2024): sensitivity, specificity, "
                              "gmean = sqrt(sens*spec), raccuracy = (sens+spec)/2 (r = negatives/positives); "
                              "roc_auc is threshold-free"),
               training_channel_masking=provenance["training_channel_masking"],
               model_type=provenance["model_type"],
               dsf=provenance["dsf"],
               test_set_evaluated=args.stage == "test", rule=calibration["rule"]))
    if args.stage == "test":
        write_json(output / "calibration.json", calibration)
    provenance["status"] = "complete"
    write_json(output / "provenance.json", provenance)
    print(f"Saved {args.stage} results to {output}", flush=True)
