"""Filter mixed acoustic detections down to bird species only.

BirdNET-Analyzer can emit non-bird classes alongside birds (human vocals,
dogs, engines, fireworks, sirens, power tools, gunshots, etc.). This module
provides a small, explicit deny-list of those non-bird BirdNET labels and
a single ``is_bird_detection`` predicate used by the output layer.
"""

from __future__ import annotations

from collections.abc import Iterable

# Tokens that, if present in the BirdNET ``scientific_name`` or
# ``common_name`` of a detection, mark that detection as non-bird.
# Matching is case-insensitive and substring-based so we tolerate small
# variations across BirdNET model releases (e.g. "Human vocal_Human vocal").
NON_BIRD_TOKENS: frozenset[str] = frozenset(
    {
        "human",         # Human vocal, Human non-vocal, Human whistle
        "homo sapiens",
        "dog",           # Dog (Canis familiaris)
        "canis",
        "engine",        # Engine
        "fireworks",
        "gun",           # Gun, Gunshot
        "power tools",
        "siren",
        "noise",
        "rain",
        "wind",
        "insect",        # generic insect class (BirdNET has a few)
        "environment",
    }
)


def is_bird_detection(species: str | None, common_name: str | None = None) -> bool:
    """Return True iff the detection looks like an actual bird species.

    A detection is considered non-bird when any deny-list token appears as a
    substring of either label. Empty / unknown labels conservatively return
    False.
    """
    if not species and not common_name:
        return False
    text = f"{species or ''} {common_name or ''}".lower()
    return not any(tok in text for tok in NON_BIRD_TOKENS)


def filter_bird_detections(detections: Iterable) -> list:
    """Return only the bird detections from a mixed iterable.

    Accepts any objects exposing ``species``/``common_name`` attributes (e.g.
    ``BirdDetection``) or mappings with the same keys.
    """
    out = []
    for d in detections:
        if isinstance(d, dict):
            sp, cn = d.get("species"), d.get("common_name")
        else:
            sp, cn = getattr(d, "species", None), getattr(d, "common_name", None)
        if is_bird_detection(sp, cn):
            out.append(d)
    return out
