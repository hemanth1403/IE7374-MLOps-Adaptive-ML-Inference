from pathlib import Path
import yaml


def test_dvc_yaml_parses():
    p = Path("dvc.yaml")
    assert p.exists(), "dvc.yaml not found (run tests from Data-Pipeline/)"
    yaml.safe_load(p.read_text(encoding="utf-8"))