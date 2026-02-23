import json
from pathlib import Path

import pytest


QUALITY = Path("data") / "reports" / "quality.json"


@pytest.mark.integration
def test_quality_invariants():
    assert QUALITY.exists(), "quality.json missing. Run: dvc repro reports"
    q = json.loads(QUALITY.read_text(encoding="utf-8"))

    # These keys match your pipeline behavior; if a key is absent, fail with a clear message.
    def get_required(key):
        assert key in q, f"quality.json missing key '{key}'"
        return q[key]

    invalid_lines = int(get_required("invalid_yolo_lines"))
    invalid_bbox = int(get_required("invalid_bbox_range"))
    missing_by_split = get_required("missing_label_files")

    assert invalid_lines == 0, f"invalid_yolo_lines should be 0, got {invalid_lines}"
    assert invalid_bbox == 0, f"invalid_bbox_range should be 0, got {invalid_bbox}"

    # missing_label_files is a dict like {"train":0,"val":0,"test":0} (or similar)
    assert isinstance(missing_by_split, dict), "missing_label_files should be a dict"
    total_missing = sum(int(v) for v in missing_by_split.values())
    assert total_missing == 0, f"missing_label_files_total should be 0, got {total_missing}"
