"""Tests for bird vs non-bird filtering."""

from dataclasses import dataclass

from models.bird_filter import filter_bird_detections, is_bird_detection


@dataclass
class _D:
    species: str
    common_name: str


def test_is_bird_detection_birds() -> None:
    assert is_bird_detection("Turdus merula", "Eurasian Blackbird")
    assert is_bird_detection("Parus major", "Great Tit")


def test_is_bird_detection_non_birds() -> None:
    assert not is_bird_detection("Human vocal", "Human vocal")
    assert not is_bird_detection("Canis familiaris", "Dog")
    assert not is_bird_detection("Engine", "Engine")
    assert not is_bird_detection("Fireworks", "Fireworks")
    assert not is_bird_detection("Power tools", "Power tools")
    assert not is_bird_detection("Siren", "Siren")
    assert not is_bird_detection("Homo sapiens", "Human")


def test_is_bird_detection_empty() -> None:
    assert not is_bird_detection(None, None)
    assert not is_bird_detection("", "")


def test_filter_bird_detections_mixed() -> None:
    mixed = [
        _D("Turdus merula", "Eurasian Blackbird"),
        _D("Human vocal", "Human vocal"),
        _D("Parus major", "Great Tit"),
        _D("Engine", "Engine"),
        _D("Canis familiaris", "Dog"),
    ]
    birds = filter_bird_detections(mixed)
    assert [b.species for b in birds] == ["Turdus merula", "Parus major"]


def test_filter_bird_detections_dicts() -> None:
    mixed = [
        {"species": "Turdus merula", "common_name": "Eurasian Blackbird"},
        {"species": "Dog", "common_name": "Dog"},
    ]
    birds = filter_bird_detections(mixed)
    assert birds == [{"species": "Turdus merula", "common_name": "Eurasian Blackbird"}]
