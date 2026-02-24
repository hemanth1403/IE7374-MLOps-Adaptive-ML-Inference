import json
from pathlib import Path

import pytest


REPORTS = Path("data") / "reports"


@pytest.mark.integration
def test_reports_exist():
    assert REPORTS.exists(), "data/reports/ not found. Run: dvc repro reports"

    expected = ["schema.json", "stats.json", "quality.json", "bias.md"]
    missing = [f for f in expected if not (REPORTS / f).exists()]
    assert not missing, f"Missing reports: {missing}. Run: dvc repro reports"


@pytest.mark.integration
def test_reports_parse():
    schema = json.loads((REPORTS / "schema.json").read_text(encoding="utf-8"))
    stats = json.loads((REPORTS / "stats.json").read_text(encoding="utf-8"))
    quality = json.loads((REPORTS / "quality.json").read_text(encoding="utf-8"))
    bias = (REPORTS / "bias.md").read_text(encoding="utf-8")

    assert isinstance(schema, dict) and schema, "schema.json should be a non-empty JSON object"
    assert isinstance(stats, dict) and stats, "stats.json should be a non-empty JSON object"
    assert isinstance(quality, dict) and quality, "quality.json should be a non-empty JSON object"
    assert "Bias / Slicing Report" in bias, "bias.md does not look like the expected report"
