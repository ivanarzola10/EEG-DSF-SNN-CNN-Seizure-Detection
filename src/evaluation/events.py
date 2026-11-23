"""Causal point alerts; no smoothing or matching may use future predictions."""
from __future__ import annotations

import itertools

import numpy as np


def validate_alarm_policy(policy):
    if policy["validation_condition"] != "complete_channels":
        raise ValueError("Alarm calibration requires complete validation recordings")
    thresholds = policy["thresholds"]
    consecutive = policy["consecutive_windows"]
    cooldown = policy["cooldown_seconds"]
    target = policy["false_alarms_per_hour_target"]
    if (1.0 not in thresholds or any(not 0 <= value <= 1 for value in thresholds)
            or not consecutive or any(type(value) is not int or value < 1 for value in consecutive)
            or not cooldown or any(not np.isfinite(value) or value < 0 for value in cooldown)
            or not np.isfinite(target) or target < 0):
        raise ValueError("Invalid alarm calibration grid. Include threshold 1.0 for a zero-alert candidate")


def alarm_times(end_times, probabilities, threshold, consecutive_windows, cooldown_seconds):
    times, scores = np.asarray(end_times), np.asarray(probabilities)
    if (times.ndim != 1 or scores.ndim != 1 or len(times) != len(scores)
            or not np.isfinite(scores).all() or not np.isfinite(times).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("Invalid prediction timestamps/probabilities")
    if len(times) > 1 and (np.diff(times) <= 0).any():
        raise ValueError("Predictions must be strictly chronological within each recording")
    if not 0 <= threshold <= 1 or consecutive_windows < 1 or cooldown_seconds < 0:
        raise ValueError("Invalid alarm rule")
    streak = 0
    active = False
    last_alarm = -np.inf
    alerts = []
    previous = -np.inf
    for time, score in zip(times, scores):
        # A missing 1-second prediction breaks consecutive evidence.
        if time - previous > 1.000001:
            streak, active = 0, False
        previous = time
        if score > threshold:
            streak += 1
        else:
            streak, active = 0, False
        if streak >= consecutive_windows and not active and time - last_alarm >= cooldown_seconds:
            alerts.append(float(time))
            last_alarm, active = time, True
    return alerts


def recording_events(end_times, probabilities, seizures, duration_seconds, rule):
    seizures = sorted(seizures, key=lambda s: s["onset_s"])
    previous_end = 0.0
    for seizure in seizures:
        onset, offset = seizure["onset_s"], seizure["offset_s"]
        if not 0 <= onset < offset <= duration_seconds or onset < previous_end:
            raise ValueError("Invalid or overlapping annotated seizure intervals")
        previous_end = offset
    alerts = alarm_times(end_times, probabilities, **rule)
    events = []
    for seizure in seizures:
        matches = [time for time in alerts if seizure["onset_s"] <= time < seizure["offset_s"]]
        detected_at = matches[0] if matches else None
        events.append({**seizure, "detected": bool(matches), "detected_at_s": detected_at,
                       "delay_s": detected_at - seizure["onset_s"] if matches else None})
    alert_rows = [dict(time_s=time, false_alarm=not any(s["onset_s"] <= time < s["offset_s"] for s in seizures))
                  for time in alerts]
    return dict(events=events, alerts=alert_rows,
                nonseizure_seconds=duration_seconds - sum(s["offset_s"] - s["onset_s"] for s in seizures))


def summarize_events(recordings):
    events = [e for r in recordings for e in r["events"]]
    false_alarms = sum(a["false_alarm"] for r in recordings for a in r["alerts"])
    nonseizure_hours = sum(r["nonseizure_seconds"] for r in recordings) / 3600
    delays = [e["delay_s"] for e in events if e["detected"]]
    return dict(seizures=len(events), detected_seizures=len(delays), missed_seizures=len(events) - len(delays),
                event_sensitivity=len(delays) / len(events) if events else None,
                false_alarms=false_alarms, nonseizure_hours=nonseizure_hours,
                false_alarms_per_hour=false_alarms / nonseizure_hours if nonseizure_hours else None,
                mean_detection_delay_s=float(np.mean(delays)) if delays else None,
                median_detection_delay_s=float(np.median(delays)) if delays else None)


def calibrate_rules(recordings, policy):
    """Maximize pooled validation sensitivity subject to a specified FA/h target.

    `recordings` contains validation predictions only. Ties favor lower delay,
    then fewer false alarms, then a higher threshold. No test outcomes are used.
    """
    candidates = []
    for threshold, consecutive, cooldown in itertools.product(
            policy["thresholds"], policy["consecutive_windows"], policy["cooldown_seconds"]):
        rule = dict(threshold=threshold, consecutive_windows=consecutive, cooldown_seconds=cooldown)
        summary = summarize_events([recording_events(**recording, rule=rule) for recording in recordings])
        candidates.append(dict(rule=rule, metrics=summary))
    feasible = [c for c in candidates if c["metrics"]["event_sensitivity"] is not None
                and c["metrics"]["false_alarms_per_hour"] is not None
                and c["metrics"]["false_alarms_per_hour"] <= policy["false_alarms_per_hour_target"]]
    if not feasible:
        raise ValueError("No alarm rule meets the validation false-alarm target")
    def ordering(candidate):
        metrics = candidate["metrics"]
        delay = metrics["mean_detection_delay_s"]
        return (-metrics["event_sensitivity"], delay if delay is not None else np.inf,
                metrics["false_alarms_per_hour"], -candidate["rule"]["threshold"])
    return min(feasible, key=ordering), candidates
