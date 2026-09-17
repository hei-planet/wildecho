"""Filter BirdNET detections down to known bird classes safely.

BirdNET can emit non-bird acoustic classes alongside birds. Filtering uses
exact normalized labels rather than substring matching so legitimate bird names
cannot be rejected just because they contain words such as "rain" or "wind".
"""

from __future__ import annotations

from collections.abc import Iterable

# Exact BirdNET labels known to represent non-bird signals. Keep this list
# conservative: unknown labels remain available in ``detected_signals`` and are
# treated as birds only when neither scientific nor common label is an exact
# match here.
NON_BIRD_LABELS: frozenset[str] = frozenset(
    {
        "human vocal",
        "human non-vocal",
        "human whistle",
        "human",
        "homo sapiens",
        "dog",
        "canis familiaris",
        "engine",
        "fireworks",
        "gun",
        "gunshot",
        "power tools",
        "siren",
        "noise",
        "rain",
        "wind",
        "insect",
        "environment",
    }
)


def _normalize_label(value: str | None) -> str:
    return " ".join((value or "").strip().casefold().split())


def is_bird_detection(species: str | None, common_name: str | None = None) -> bool:
    """Return True unless either complete label is a known non-bird class."""
    species_label = _normalize_label(species)
    common_label = _normalize_label(common_name)
    if not species_label and not common_label:
        return False
    return species_label not in NON_BIRD_LABELS and common_label not in NON_BIRD_LABELS


def filter_bird_detections(detections: Iterable) -> list:
    """Return only bird detections from objects or mappings."""
    out = []
    for detection in detections:
        if isinstance(detection, dict):
            species = detection.get("species")
            common_name = detection.get("common_name")
        else:
            species = getattr(detection, "species", None)
            common_name = getattr(detection, "common_name", None)
        if is_bird_detection(species, common_name):
            out.append(detection)
    return out
