import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.channel_masks import (training_mask, make_mask_manifest, validate_mask_manifest,
                                    RecordingMaskedDataset)
from src.evaluation.events import alarm_times, recording_events, summarize_events, calibrate_rules


def experiment():
    from pathlib import Path
    return json.loads(Path("configs/experiment.json").read_text())


def test_training_masks_are_seeded_include_clean_and_have_correct_channel_counts():
    x = torch.ones(128, 18, 8)
    policy = experiment()
    first, keep = training_mask(x, torch.Generator().manual_seed(9), policy["missing_fractions"], policy["training_mask_probabilities"])
    second, _ = training_mask(x, torch.Generator().manual_seed(9), policy["missing_fractions"], policy["training_mask_probabilities"])
    torch.testing.assert_close(first, second)
    assert set((18 - keep.sum(1)).tolist()) == {0, 5, 9, 14}
    torch.testing.assert_close(first, keep[:, :, None].expand_as(x).float())
    assert torch.equal(x, torch.ones_like(x))


def test_recording_masks_are_shared_fixed_and_checked():
    manifest = dict(data_config={"montage": {"derivations": [str(i) for i in range(18)]}},
                    included=[dict(split="test", file_id="record")])
    policy = experiment()
    saved = make_mask_manifest(manifest, policy, "hash")
    assert saved == make_mask_manifest(manifest, policy, "hash")
    assert len(saved["masks"]) == 16
    data = TensorDataset(torch.ones(3, 18, 5), torch.zeros(3, dtype=torch.long))
    data.df = pd.DataFrame(dict(file_id=["record"] * 3))
    dataset = RecordingMaskedDataset(data, saved, "test", .5, 1)
    torch.testing.assert_close(dataset[0][0], dataset[2][0])
    assert (dataset[0][0].sum(1) == 0).sum() == 9
    saved["masks"][-1]["missing_indices"] = []
    with pytest.raises(ValueError, match="differs"):
        validate_mask_manifest(saved, manifest, policy, "hash")


def test_alarm_processing_is_causal_and_resets_on_prediction_gaps():
    rule = dict(threshold=.5, consecutive_windows=2, cooldown_seconds=0)
    times = np.arange(4, 10)
    scores = [.8, .9, .2, .8, .9, .9]
    full = alarm_times(times, scores, **rule)
    prefix = alarm_times(times[:3], scores[:3], **rule)
    assert full == [5., 8.]
    assert prefix == [time for time in full if time <= times[2]]
    assert alarm_times([4, 6], [.9, .9], **rule) == []
    assert alarm_times(times, scores, **{**rule, "cooldown_seconds": 10}) == [5.]
    assert alarm_times(times, scores, **{**rule, "threshold": 1.}) == []


def test_event_matching_delay_false_alarm_denominator_and_misses():
    rule = dict(threshold=.5, consecutive_windows=1, cooldown_seconds=0)
    seizures = [dict(event_id="a", onset_s=6, offset_s=9), dict(event_id="b", onset_s=11, offset_s=13)]
    # Alerts at 4 (false), 7 (detect a), 9 (false: offset is exclusive).
    result = recording_events(np.arange(4, 14), [.9, .1, .1, .9, .1, .9, .1, .1, .1, .1], seizures, 20, rule)
    summary = summarize_events([result])
    assert summary["event_sensitivity"] == .5
    assert summary["false_alarms"] == 2
    assert summary["false_alarms_per_hour"] == pytest.approx(2 / (15 / 3600))
    assert summary["mean_detection_delay_s"] == 1
    assert summary["missed_seizures"] == 1
    assert result["events"][1]["delay_s"] is None


def test_calibration_selects_validation_rule_under_false_alarm_target():
    times = np.arange(4, 20)
    scores = np.where((times >= 8) & (times < 12), .99, .01)
    recordings = [dict(end_times=times, probabilities=scores, duration_seconds=20,
                       seizures=[dict(event_id="a", onset_s=8, offset_s=12)])]
    policy = dict(thresholds=[.5, .9, 1.], consecutive_windows=[1, 2], cooldown_seconds=[0],
                  false_alarms_per_hour_target=1.)
    selected, candidates = calibrate_rules(recordings, policy)
    assert len(candidates) == 6
    assert selected["rule"]["threshold"] == .9
    assert selected["metrics"]["event_sensitivity"] == 1
    assert selected["metrics"]["mean_detection_delay_s"] == 0


def test_patient_evaluation_combines_alias_case_folders():
    from types import SimpleNamespace
    from src.evaluation.pipeline import patient_results
    dataset = SimpleNamespace(df=pd.DataFrame(dict(patient_id=["chb01", "chb01", "chb21", "chb21"])))
    records = [dict(patient_id=p, subject_id="chb01", file_id=p + "_01",
                    end_times=np.array([4., 5.]), probabilities=np.array([.1, .9]), duration_seconds=6.,
                    seizures=[dict(event_id=p + "#0", onset_s=5., offset_s=6.)]) for p in ("chb01", "chb21")]
    patients, events, _ = patient_results(dataset, np.array([0, 1, 0, 1]), np.array([.1, .9, .1, .9]),
                                         records, dict(threshold=.5, consecutive_windows=1, cooldown_seconds=0), 0., 0)
    assert len(patients) == 1
    assert patients[0]["subject_id"] == "chb01"
    assert patients[0]["seizures"] == patients[0]["detected_seizures"] == 2
    assert len(events) == 2
