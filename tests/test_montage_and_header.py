from src.data.edf_io import parse_edf_header, patient_id_from_file_id
from src.data.montage import match_montage


def make_edf_header(labels, samples_per_record=256, n_records=3600, record_s=1):
    def pad(value, width):
        return str(value).ljust(width)[:width].encode("ascii")

    ns = len(labels)
    fixed = (pad(0, 8) + pad("X", 80) + pad("X", 80) + pad("01.01.01", 8) + pad("11.42.54", 8)
             + pad(256 * (ns + 1), 8) + pad("", 44) + pad(n_records, 8) + pad(record_s, 8) + pad(ns, 4))
    fields = [(labels, 16), (["" ] * ns, 80), (["uV"] * ns, 8), (["-800"] * ns, 8), (["800"] * ns, 8),
              (["-32768"] * ns, 8), (["32767"] * ns, 8), ([""] * ns, 80), ([samples_per_record] * ns, 8),
              ([""] * ns, 32)]
    signal = b"".join(pad(v, w) for values, w in fields for v in values)
    return fixed + signal


def test_header_parser_reads_exact_labels_and_rates():
    raw = make_edf_header(["FP1-F7", "T8-P8", "-", "T8-P8"])
    h = parse_edf_header(raw)
    assert h.labels == ("FP1-F7", "T8-P8", "-", "T8-P8")
    assert h.sampling_rates == (256.0,) * 4
    assert h.duration_s == 3600
    assert h.header_bytes == len(raw)


def test_montage_first_duplicate_and_missing():
    labels = ["fp1-f7 ", "T8-P8", "-", "T8-P8", "ECG"]
    m = match_montage(labels, ["FP1-F7", "T8-P8"])
    assert m.complete and m.indices == (0, 1)
    assert m.duplicates == {"T8-P8": (1, 3)}
    assert m.extras == ("ECG",) and m.n_dummy == 1

    m2 = match_montage(labels, ["FP1-F7", "CZ-PZ"])
    assert not m2.complete and m2.missing == ("CZ-PZ",) and m2.indices == ()


def test_referential_labels_are_not_silently_converted():
    m = match_montage(["FP1-CS2", "F7-CS2"], ["FP1-F7"])
    assert m.missing == ("FP1-F7",)


def test_patient_ids():
    assert patient_id_from_file_id("chb02_16+") == "chb02"
