"""Window overlap with the union of recording-relative seizure intervals."""

from .annotations import Seizure


def seizure_overlap_s(window_start_s: float, window_end_s: float, seizures: list[Seizure] | tuple[Seizure, ...]) -> float:
    """Return overlapping seconds, counting intersecting seizure intervals once."""
    intervals = sorted(
        (max(window_start_s, seizure.onset_s), min(window_end_s, seizure.offset_s))
        for seizure in seizures
        if seizure.onset_s < window_end_s and seizure.offset_s > window_start_s
    )
    total = 0.0
    last_end = window_start_s
    for start, end in intervals:
        total += max(0.0, end - max(start, last_end))
        last_end = max(last_end, end)
    return total
