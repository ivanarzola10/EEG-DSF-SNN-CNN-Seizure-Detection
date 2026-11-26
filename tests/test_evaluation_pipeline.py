"""Exercise calibration and all missing-channel test conditions on synthetic EEG."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.pipeline import run, fingerprint
from src.models.cnn import ShallowEEGCNN


def make_experiment(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    channels = [f"channel{i}" for i in range(18)]
    splits = dict(train=["train_patient"], val=["val_patient"], test=["test_patient"])
    included, metadata, events = [], [], []
    rng = np.random.default_rng(9)
    for split, patients in splits.items():
        patient = patients[0]
        file_id = patient + "_01"
        np.save(root / f"{file_id}.npy", rng.normal(size=(18, 3072)).astype(np.float32))
        positives = 0
        for start in range(0, 3072 - 1024 + 1, 256):
            start_s, end_s = start / 256, (start + 1024) / 256
            label = int(max(0, min(end_s, 9) - max(start_s, 6)) >= 2)
            positives += label
            metadata.append(dict(patient_id=patient, file_id=file_id, split=split,
                                 start_sample=start, start_s=start_s, end_s=end_s, label=label,
                                 continuous_path=f"{file_id}.npy"))
        included.append(dict(file_id=file_id, split=split, patient_id=patient, samples=3072,
                             windows=9, positive_windows=positives))
        events.append(dict(file_id=file_id, event_id=file_id + "#0", onset_s=6, offset_s=9))
    manifest = dict(status="complete", protocol="proposal_unfiltered_v1", splits=splits,
                    data_config=dict(dataset=dict(same_subject={}), montage=dict(derivations=channels)),
                    included=included, normalization_records=["train_patient_01"],
                    split_counts={r["split"]: dict(windows=r["windows"], positive_windows=r["positive_windows"])
                                  for r in included})
    (root / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(metadata).to_csv(root / "metadata.csv", index=False)
    pd.DataFrame(events).to_csv(root / "seizures.csv", index=False)
    np.savez(root / "normalization.npz", mean=np.zeros((18, 1)), std=np.ones((18, 1)),
             channels=np.array(channels), train_patients=np.array(splits["train"]),
             records=np.array(["train_patient_01"]), source="continuous_unfiltered_recordings", protocol="proposal_unfiltered_v1")
    experiment = json.loads(Path("configs/experiment.json").read_text())
    experiment["evaluation_repeats"] = 2
    experiment["alarm_calibration"].update(thresholds=[.5, 1.], consecutive_windows=[1], cooldown_seconds=[0])
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(experiment))
    model_config = dict(n_temporal_filters=2, temporal_kernel_size=5, pool_size=10, pool_stride=5, dropout=0.)
    torch.manual_seed(9)
    model = ShallowEEGCNN(**model_config)
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(model_state_dict=model.state_dict(), config=dict(model=model_config, channel_masking="mixed", experiment=experiment),
                    provenance=dict(fingerprints={p: fingerprint(root / p) for p in ("manifest.json", "metadata.csv", "normalization.npz", "seizures.csv")})), checkpoint)
    return SimpleNamespace(data_root=root, experiment_config=config_path, checkpoint=checkpoint,
                           mask_manifest=tmp_path / "masks.json", device="cpu", threads=0, workers=0,
                           batch_size=4, log_every=0, output=tmp_path / "calibration", stage="calibrate", calibration=None)


def test_pipeline_freezes_validation_rule_and_scores_all_test_masks(tmp_path, monkeypatch):
    import src.evaluation.pipeline as pipeline
    arguments = make_experiment(tmp_path)
    original = pipeline.WindowDataset
    loaded = []
    def tracked_dataset(*args, **kwargs):
        loaded.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "WindowDataset", tracked_dataset)
    run(arguments)
    assert loaded == [["val_patient"]]
    calibration_path = arguments.output / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    assert calibration["split"] == "val" and not calibration["test_set_evaluated"]
    arguments.stage = "test"
    arguments.calibration = calibration_path
    arguments.output = tmp_path / "test_results"
    run(arguments)
    assert loaded[-1] == ["test_patient"]
    result = json.loads((arguments.output / "summary.json").read_text())
    assert result["test_set_evaluated"] and result["rule"] == calibration["rule"]
    frame = pd.read_csv(arguments.output / "patient_metrics.csv")
    assert len(frame) == 7  # complete once; three loss levels, two subsets each.
    assert set(frame.windows) == {9}
    assert set(frame.seizures) == {1}
    assert len(pd.read_csv(arguments.output / "seizure_events.csv")) == 7
    assert result["timing"]["batch_size"] == 4
    assert result["timing"]["parameters"] > 0
    assert len(list((arguments.output / "predictions").glob("*.npz"))) == 7
    # A different model checkpoint cannot borrow the selected rule.
    calibration["checkpoint_sha256"] = "different_model"
    calibration_path.write_text(json.dumps(calibration))
    arguments.output = tmp_path / "must_not_evaluate"
    with pytest.raises(ValueError, match="Calibration does not match"):
        run(arguments)
    assert not arguments.output.exists()
