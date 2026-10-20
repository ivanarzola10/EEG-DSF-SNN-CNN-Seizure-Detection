"""Parse CHB-MIT ``chbXX-summary.txt`` files into per-file seizure intervals.

Handles both label styles seen in the dataset ("Seizure Start Time:" and
"Seizure 3 Start Time:"), and the "Channels in EDF Files:" / "Channels changed:" blocks,
which apply to every file listed after them until the next block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FILE = re.compile(r"^File Name:\s*(\S+)")
_FILE_START = re.compile(r"^File Start Time:\s*(\S+)")
_FILE_END = re.compile(r"^File End Time:\s*(\S+)")
_N_SEIZ = re.compile(r"^Number of Seizures in File:\s*(\d+)")
_SEIZ = re.compile(r"^Seizure(?:\s+(\d+))?\s+(Start|End)\s+Time:\s*(\d+)\s*sec", re.IGNORECASE)
_CHANNEL = re.compile(r"^Channel\s+(\d+):\s*(.*)$")
_CHANNEL_BLOCK = re.compile(r"^Channels (in EDF Files|changed):")


@dataclass(frozen=True)
class Seizure:
    file_id: str
    index: int          # 0-based order within the file
    onset_s: float
    offset_s: float

    @property
    def event_id(self) -> str:
        return f"{self.file_id}#{self.index}"

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


@dataclass
class SummaryEntry:
    file_id: str
    start_time: str | None = None
    end_time: str | None = None
    n_seizures_declared: int | None = None
    seizures: list[Seizure] = field(default_factory=list)
    channels: tuple[str, ...] = ()     # channel list in effect for this file per the summary


def parse_summary(text: str) -> dict[str, SummaryEntry]:
    entries: dict[str, SummaryEntry] = {}
    current: SummaryEntry | None = None
    channels: list[str] = []
    in_channel_block = False
    # [start, end] pairs in file order. Each End closes the open Start; the seizure number in
    # the label is ignored because the source has typos (chb09_08: "Seizure 1 End" for seizure 2).
    pending: list[list[int | None]] = []

    def close_file() -> None:
        if current is None:
            return
        seizures = []
        for k, (start, end) in enumerate(pending):
            if end is None:
                raise ValueError(f"{current.file_id}: seizure {k + 1} lacks an end")
            if end <= start:
                raise ValueError(f"{current.file_id}: seizure {k + 1} ends before it starts")
            seizures.append(Seizure(current.file_id, k, float(start), float(end)))
        if current.n_seizures_declared is not None and len(seizures) != current.n_seizures_declared:
            raise ValueError(
                f"{current.file_id}: declares {current.n_seizures_declared} seizures, found {len(seizures)}"
            )
        for a, b in zip(seizures, seizures[1:]):
            if b.onset_s < a.offset_s:
                raise ValueError(f"{current.file_id}: overlapping seizures {a} / {b}")
        current.seizures = seizures
        entries[current.file_id] = current

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _CHANNEL_BLOCK.match(line):
            close_file()
            current, pending = None, []
            channels, in_channel_block = [], True
            continue
        if in_channel_block and (m := _CHANNEL.match(line)):
            channels.append(m.group(2).strip())
            continue
        if m := _FILE.match(line):
            close_file()
            in_channel_block = False
            file_id = m.group(1).removesuffix(".edf")
            if file_id in entries:
                raise ValueError(f"{file_id} listed twice in summary")
            current, pending = SummaryEntry(file_id, channels=tuple(channels)), []
            continue
        if current is None:
            continue
        if m := _FILE_START.match(line):
            current.start_time = m.group(1)
        elif m := _FILE_END.match(line):
            current.end_time = m.group(1)
        elif m := _N_SEIZ.match(line):
            current.n_seizures_declared = int(m.group(1))
        elif m := _SEIZ.match(line):
            kind, seconds = m.group(2).capitalize(), int(m.group(3))
            if kind == "Start":
                if pending and pending[-1][1] is None:
                    raise ValueError(f"{current.file_id}: seizure {len(pending)} lacks an end")
                pending.append([seconds, None])
            elif not pending or pending[-1][1] is not None:
                raise ValueError(f"{current.file_id}: seizure end without a start")
            else:
                pending[-1][1] = seconds
    close_file()
    return entries


def load_summary(path: str | Path) -> dict[str, SummaryEntry]:
    return parse_summary(Path(path).read_text(encoding="utf-8", errors="replace"))


def clock_to_seconds(clock: str) -> int:
    """'24:44:29' -> 89069. CHB-MIT clock times can exceed 24 h or drop the leading zero."""
    h, m, s = (int(x) for x in clock.split(":"))
    return h * 3600 + m * 60 + s


def summary_duration_s(entry: SummaryEntry) -> int | None:
    """Recording length implied by the summary clock times (mod 24 h), if present."""
    if not entry.start_time or not entry.end_time:
        return None
    return (clock_to_seconds(entry.end_time) - clock_to_seconds(entry.start_time)) % 86400
