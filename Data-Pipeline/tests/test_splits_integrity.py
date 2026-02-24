from pathlib import Path

import pytest


SPLITS = Path("data") / "splits"


def _load(p: Path) -> list[str]:
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.integration
def test_splits_exist_and_nonempty():
    for name in ["train.txt", "val.txt", "test.txt"]:
        p = SPLITS / name
        assert p.exists(), f"Missing {p}. Run: dvc repro splits"
        lines = _load(p)
        assert len(lines) > 0, f"{name} is empty"


@pytest.mark.integration
def test_splits_no_overlap():
    train = set(_load(SPLITS / "train.txt"))
    val = set(_load(SPLITS / "val.txt"))
    test = set(_load(SPLITS / "test.txt"))

    assert train.isdisjoint(val), "Overlap found between train and val"
    assert train.isdisjoint(test), "Overlap found between train and test"
    assert val.isdisjoint(test), "Overlap found between val and test"
