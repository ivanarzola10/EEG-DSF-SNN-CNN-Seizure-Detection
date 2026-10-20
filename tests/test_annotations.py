import pytest

from src.data.annotations import clock_to_seconds, parse_summary, summary_duration_s

SUMMARY = """Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: F7-T7

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

Channels changed:
Channel 1: -
Channel 2: FP1-F7

File Name: chb06_01.edf
File Start Time: 23:44:29
File End Time: 24:44:29
Number of Seizures in File: 3
Seizure 1 Start Time: 1724 seconds
Seizure 1 End Time: 1738 seconds
Seizure 2 Start Time:  7461 seconds
Seizure 2 End Time: 7476 seconds
Seizure 3 Start Time: 13525 seconds
Seizure 3 End Time: 13540 seconds
"""


def test_parses_both_seizure_formats_and_channel_blocks():
    entries = parse_summary(SUMMARY)
    assert list(entries) == ["chb01_01", "chb01_03", "chb06_01"]
    assert entries["chb01_01"].seizures == []
    (sz,) = entries["chb01_03"].seizures
    assert (sz.onset_s, sz.offset_s, sz.event_id) == (2996, 3036, "chb01_03#0")
    assert [(s.onset_s, s.offset_s) for s in entries["chb06_01"].seizures] == [
        (1724, 1738), (7461, 7476), (13525, 13540)]
    assert entries["chb01_03"].channels == ("FP1-F7", "F7-T7")
    assert entries["chb06_01"].channels == ("-", "FP1-F7")


def test_declared_count_mismatch_raises():
    bad = SUMMARY.replace("Number of Seizures in File: 1", "Number of Seizures in File: 2")
    with pytest.raises(ValueError, match="declares 2"):
        parse_summary(bad)


def test_inverted_interval_raises():
    bad = SUMMARY.replace("Seizure End Time: 3036", "Seizure End Time: 2990")
    with pytest.raises(ValueError, match="ends before"):
        parse_summary(bad)


def test_mislabelled_seizure_number_pairs_in_order():
    # chb09_08 in the real summary labels its second seizure's end "Seizure 1 End Time".
    text = SUMMARY.replace("Seizure 2 End Time: 7476", "Seizure 1 End Time: 7476")
    assert [(s.onset_s, s.offset_s) for s in parse_summary(text)["chb06_01"].seizures] == [
        (1724, 1738), (7461, 7476), (13525, 13540)]


def test_end_without_start_raises():
    bad = SUMMARY.replace("Seizure Start Time: 2996 seconds\n", "")
    with pytest.raises(ValueError, match="without a start"):
        parse_summary(bad)


def test_clock_times_past_midnight():
    assert clock_to_seconds("24:44:29") == 24 * 3600 + 44 * 60 + 29
    entries = parse_summary(SUMMARY)
    assert summary_duration_s(entries["chb06_01"]) == 3600
    assert summary_duration_s(entries["chb01_01"]) == 3600
