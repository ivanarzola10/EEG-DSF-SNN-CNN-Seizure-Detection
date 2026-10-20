"""Bipolar montage matching.

Harmonization policy (deliberately minimal): labels are compared after ``strip().upper()``.
No other renaming, re-referencing or channel synthesis is performed. A derivation present
more than once maps to its first occurrence, and the duplicates are reported.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

DUMMY_LABELS = {"-", ".", ""}


def canonical(label: str) -> str:
    return label.strip().upper()


@dataclass(frozen=True)
class MontageMatch:
    indices: tuple[int, ...]              # header index of each found derivation, config order
    missing: tuple[str, ...]              # required derivations not found
    duplicates: dict[str, tuple[int, ...]]  # required derivation -> all header indices (len > 1)
    extras: tuple[str, ...]               # non-dummy labels not in the montage
    n_dummy: int                          # placeholder channels such as '-' or '.'

    @property
    def complete(self) -> bool:
        return not self.missing


def match_montage(labels: list[str] | tuple[str, ...], derivations: list[str]) -> MontageMatch:
    positions: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        positions[canonical(label)].append(i)

    wanted = [canonical(d) for d in derivations]
    indices = tuple(positions[d][0] for d in wanted if positions.get(d))
    missing = tuple(d for d, w in zip(derivations, wanted) if not positions.get(w))
    duplicates = {d: tuple(positions[w]) for d, w in zip(derivations, wanted) if len(positions.get(w, [])) > 1}
    wanted_set = set(wanted)
    extras = tuple(l for l in labels if canonical(l) not in wanted_set and canonical(l) not in DUMMY_LABELS)
    n_dummy = sum(canonical(l) in DUMMY_LABELS for l in labels)
    return MontageMatch(indices if not missing else (), missing, duplicates, extras, n_dummy)
