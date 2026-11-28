"""Reproducible CNN fitting; held-out test data never enters model selection."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from src.data.channel_masks import training_mask


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


PAPER_THRESHOLD = 0.5  # Zhang et al. 2024 classify a window as seizure if p > 0.5


def binary_metrics(labels, probabilities):
    """Window-level metrics. Threshold-free: PR-AUC, average precision, ROC-AUC.
    At the paper's fixed threshold (p > 0.5, independent of the calibrated alarm rule):
    sensitivity, specificity, GMean = sqrt(sens * spec) and RAccuracy = (sens + spec) / 2,
    which is the paper's r-weighted accuracy with r = negatives / positives (balanced
    accuracy). Undefined values are None (e.g. sensitivity with no seizure windows)."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected nonempty binary labels")
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Expected one finite probability per label")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be between zero and one")
    metrics = dict(windows=len(labels), positives=int(labels.sum()), prevalence=float(labels.mean()),
                   pr_auc=None, average_precision=None, roc_auc=None)
    if len(np.unique(labels)) == 2:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        metrics.update(pr_auc=float(auc(recall, precision)),
                       average_precision=float(average_precision_score(labels, probabilities)),
                       roc_auc=float(roc_auc_score(labels, probabilities)))
    predicted = probabilities > PAPER_THRESHOLD
    positive = labels == 1
    tp, fn = int((predicted & positive).sum()), int((~predicted & positive).sum())
    tn, fp = int((~predicted & ~positive).sum()), int((predicted & ~positive).sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both = sensitivity is not None and specificity is not None
    metrics.update(decision_threshold=PAPER_THRESHOLD, tp=tp, fp=fp, tn=tn, fn=fn,
                   sensitivity=sensitivity, specificity=specificity,
                   gmean=math.sqrt(sensitivity * specificity) if both else None,
                   raccuracy=(sensitivity + specificity) / 2 if both else None)
    return metrics


def class_weights(labels):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if len(counts) != 2 or not np.all(counts > 0):
        raise ValueError("Training must contain seizure and non-seizure windows")
    return torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)


def evaluate(model, loader, device, log_every=0):
    model.eval()
    labels, probabilities = [], []
    total_loss = 0.0
    total = 0
    with torch.inference_mode():
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            total_loss += loss.item() * len(y)
            total += len(y)
            labels.append(y.cpu().numpy())
            probabilities.append(logits.softmax(1)[:, 1].cpu().numpy())
            if log_every and step % log_every == 0:
                print(f"  validation batches {step}/{len(loader)}", flush=True)
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    return {**binary_metrics(y, p), "loss": total_loss / total}, y, p


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(model, train_ds, val_ds, output, config, provenance, on_epoch_end=None):
    """Train one candidate and select the epoch by full-validation PR-AUC.

    ``on_epoch_end(epoch, model)``, if given, runs after each validation pass; a dict it
    returns is stored in that epoch's history row (the SNN logs firing rates this way).
    It must not change the model or the selection.

    Class imbalance is handled only by weighted loss; samples are uniformly
    shuffled without replacement. A sample budget, if set, defines a shorter
    epoch. Validation always uses all windows in their original distribution.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    device = choose_device(config["device"])
    model = model.to(device)
    generator = torch.Generator().manual_seed(config["seed"])
    worker_generator = torch.Generator().manual_seed(config["seed"] + 1)
    mask_generator = torch.Generator().manual_seed(config["seed"] + 2)
    train_labels = train_ds.df.label.to_numpy()
    if len(np.unique(val_ds.df.label)) != 2:
        raise ValueError("Validation must contain both classes to select by PR-AUC")
    budget = config["samples_per_epoch"] or len(train_ds)
    budget = min(budget, len(train_ds))
    sampler = RandomSampler(train_ds, replacement=False, num_samples=budget, generator=generator)
    common = dict(num_workers=config["workers"], worker_init_fn=seed_worker,
                  generator=worker_generator)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, **common)
    val_loader = DataLoader(val_ds, batch_size=config["validation_batch_size"], shuffle=False, **common)
    weights = class_weights(train_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    best_score = -math.inf
    best_epoch = 0
    history = []
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters())}; "
          f"training windows/epoch={budget}; validation windows={len(val_ds)}", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            if config.get("channel_masking", "none") == "mixed":
                x, _ = training_mask(x, mask_generator, config["experiment"]["missing_fractions"],
                                     config["experiment"]["training_mask_probabilities"])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(y)
            seen += len(y)
            if config["log_every"] and step % config["log_every"] == 0:
                print(f"epoch {epoch}: training batches {step}/{len(train_loader)}; loss={total_loss / seen:.5f}", flush=True)
        metrics, labels, probabilities = evaluate(model, val_loader, device, config["log_every"])
        row = dict(epoch=epoch, train_loss=total_loss / seen, training_windows=seen,
                   validation=metrics, seconds=time.perf_counter() - start)
        if on_epoch_end is not None:
            row.update(on_epoch_end(epoch, model) or {})
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        improved = metrics["pr_auc"] > best_score
        if improved:
            best_score, best_epoch = metrics["pr_auc"], epoch
        checkpoint = dict(epoch=epoch, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), config=config,
                          best_epoch=best_epoch, best_validation_pr_auc=best_score,
                          validation=metrics, provenance=provenance,
                          training_class_weights=weights.detach().cpu(),
                          torch_rng_state=torch.get_rng_state(), sampler_rng_state=generator.get_state(),
                          mask_rng_state=mask_generator.get_state())
        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            np.savez_compressed(output / "best_validation_predictions.npz", labels=labels, probabilities=probabilities)
            if "patient_id" in val_ds.df:
                patients = val_ds.df.patient_id.to_numpy()
                per_patient = {str(p): binary_metrics(labels[patients == p], probabilities[patients == p])
                               for p in np.unique(patients)}
                (output / "best_validation_by_patient.json").write_text(json.dumps(per_patient, indent=2) + "\n")
        print(f"epoch {epoch}: train_loss={row['train_loss']:.5f}; val_loss={metrics['loss']:.5f}; "
              f"PR-AUC={metrics['pr_auc']:.6f}; AP={metrics['average_precision']:.6f}; "
              f"seconds={row['seconds']:.1f}; best_epoch={best_epoch}", flush=True)
        if config["patience"] and epoch - best_epoch >= config["patience"]:
            print(f"Early stopping after epoch {epoch}", flush=True)
            break
    summary = dict(best_epoch=best_epoch, best_validation_pr_auc=best_score,
                   epochs_completed=len(history), checkpoint=str(output / "best.pt"),
                   device=str(device), parameters=sum(p.numel() for p in model.parameters()),
                   test_evaluated=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
