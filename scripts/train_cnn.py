"""Prepare, train, calibrate and evaluate the standalone proposal CNN.

Run from the repository root: python -m scripts.train_cnn --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from scripts.prepare_dataset import CACHE, dataset_spec, ensure_prepared, resolve_inputs, smoke_sources
from src.data.channel_masks import validate_experiment
from src.data.torch_dataset import WindowDataset
from src.evaluation.events import validate_alarm_policy
from src.evaluation.pipeline import fingerprint, run, write_json
from src.models.dsf import PRESETS, dsf_spec
from src.models.registry import build_model
from src.training.cnn import fit, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=CACHE, help="Preferred prepared cache directory")
    parser.add_argument("--smoke", action="store_true", help="Exercise the pipeline on three short real EDF excerpts")
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--channel-masking", choices=["mixed", "none"], default="mixed",
                        help="Default: uniform complete/25/50/75%% missing-channel augmentation; none is an ablation")
    parser.add_argument("--dsf", choices=["none", *PRESETS], default="none",
                        help="Dynamic Spatial Filtering front end: dsfd (log-variance) or dsfm_st "
                             "(matrix-log covariance + soft threshold); none = plain CNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="Epochs without improved validation PR-AUC; 0 disables")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Uniform training-window budget per epoch; 0 uses all training windows")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch default")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "validation_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("patience", "samples_per_epoch", "workers", "threads", "log_every", "seed"):
        if getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0:
        parser.error("Invalid dropout, learning rate, weight decay or gradient clip")
    return args


def main():
    args = parse_args()
    experiment = json.loads(args.experiment_config.read_text())
    validate_experiment(experiment)
    validate_alarm_policy(experiment["alarm_calibration"])
    cfg, cohort, sources, unavailable = resolve_inputs(args.raw_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = "cnn" if args.dsf == "none" else f"cnn_{args.dsf}"
    output = (Path("runs") / (f"{name}_smoke" if args.smoke else name) / timestamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.data_root.resolve()
    if args.smoke:
        cache = output / "prepared"
        sources = smoke_sources(sources, cfg["dataset"]["sampling_rate_hz"])
        args.epochs = 1
        args.threads = args.threads or 2
    spec = dataset_spec(cfg, cohort, sources, unavailable, args.smoke)
    root, manifest = ensure_prepared(cache, spec, sources)
    print(f"Using prepared dataset {root}", flush=True)
    datasets = {split: WindowDataset(root / "metadata.csv", manifest["splits"][split], root / "normalization.npz")
                for split in ("train", "val")}
    for split, dataset in datasets.items():
        if set(dataset.df.split) != {split} or set(dataset.df.label) != {0, 1}:
            raise ValueError(f"Invalid {split} membership or missing label class")
        counts = manifest["split_counts"][split]
        if len(dataset) != counts["windows"] or int(dataset.df.label.sum()) != counts["positive_windows"]:
            raise ValueError(f"{split} counts differ from prepared manifest")
    if args.threads:
        torch.set_num_threads(args.threads)
    provenance = dict(data_root=str(root), protocol=manifest["protocol"], smoke=args.smoke,
                      fingerprints={name: fingerprint(root / name) for name in ("manifest.json", "metadata.csv", "normalization.npz", "seizures.csv")},
                      splits=manifest["splits"], channels=manifest["data_config"]["montage"]["derivations"],
                      normalization=dict(mean=datasets["train"].mean.tolist(), std=datasets["train"].std.tolist()),
                      versions={name: version(name) for name in ("torch", "braindecode", "numpy", "scikit-learn")},
                      source_sha256={str(path): fingerprint(path) for path in (
                          *sorted(Path("src").rglob("*.py")), Path(__file__),
                          Path("scripts/prepare_dataset.py"), Path("configs/recording_inventory.csv"))},
                      selection_metric="trapezoidal precision-recall AUC; average precision reported separately",
                      reproducibility="fixed RNG seeds; deterministic algorithms requested with warnings; bitwise identity across hardware is not guaranteed",
                      channel_masking=args.channel_masking, experiment=experiment,
                      dsf=dsf_spec(args.dsf), test_evaluated=False)
    write_json(output / "provenance.json", provenance)
    experiment_path = output / "experiment.json"
    write_json(experiment_path, experiment)
    shutil.copy2(root / "manifest.json", output / "dataset_manifest.json")
    shutil.copy2(root / "normalization.npz", output / "normalization.npz")
    filters = [2] if args.smoke else [20, 40]
    settings = [(count, kernel, pool, stride)
                for count, kernel, (pool, stride) in itertools.product(filters, [25, 51], [(75, 15), (150, 30)])]
    config = {k: v for k, v in vars(args).items() if k not in ("raw_dir", "data_root", "experiment_config")}
    config.update(experiment=experiment, imbalance_policy="inverse-frequency weighted loss only",
                  model_type="cnn", dsf=dsf_spec(args.dsf))
    write_json(output / "config.json", dict(config, candidates=settings))
    results = []
    best = None
    for trial, (filters, kernel, pool, stride) in enumerate(settings):
        seed_everything(args.seed)
        model_config = dict(n_temporal_filters=filters, temporal_kernel_size=kernel,
                            pool_size=pool, pool_stride=stride, dropout=args.dropout,
                            n_channels=18, n_times=1024, n_classes=2)
        print(f"Candidate {trial + 1}/{len(settings)}: {model_config}", flush=True)
        candidate = dict(config, model=model_config)
        summary = fit(build_model(candidate), datasets["train"], datasets["val"],
                      output / f"candidate_{trial:02}", candidate, provenance)
        results.append({**summary, "model": model_config})
        if best is None or summary["best_validation_pr_auc"] > best["best_validation_pr_auc"]:
            best = results[-1]
            shutil.copy2(summary["checkpoint"], output / "best.pt")
        selection = dict(best=best, candidates=results, test_evaluated=False,
                         hyperparameter_search_completed=len(results) == len(settings))
        write_json(output / "selection.json", selection)
    print(f"Selected checkpoint: {output / 'best.pt'}; validation PR-AUC={best['best_validation_pr_auc']:.6f}", flush=True)
    evaluation = dict(data_root=root, experiment_config=experiment_path, checkpoint=output / "best.pt",
                      mask_manifest=output / "evaluation_masks.json", device=args.device, threads=args.threads,
                      batch_size=args.batch_size, workers=args.workers, log_every=args.log_every)
    calibration = output / "validation" / "calibration.json"
    run(SimpleNamespace(**evaluation, stage="calibrate", output=calibration.parent, calibration=None))
    run(SimpleNamespace(**evaluation, stage="test", output=output / "test", calibration=calibration))
    write_json(output / "selection.json", dict(selection, test_evaluated=True))
    print(f"Completed {name} {'smoke' if args.smoke else 'final'} run at {output}", flush=True)


if __name__ == "__main__":
    main()
