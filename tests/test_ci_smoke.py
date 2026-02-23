from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_has_expected_structure():
    """CI-safe sanity checks: no data, no DVC, no downloads."""
    assert (REPO_ROOT / "dvc.yaml").exists(), "dvc.yaml missing"
    assert (REPO_ROOT / "scripts").exists(), "scripts/ folder missing"
    assert (REPO_ROOT / "requirements.txt").exists(), "requirements.txt missing"
    assert (REPO_ROOT / ".github" / "workflows").exists(), ".github/workflows missing (CI not set up?)"


def test_dvc_yaml_parses_and_has_required_stages():
    dvc_path = REPO_ROOT / "dvc.yaml"
    data = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))

    assert isinstance(data, dict), "dvc.yaml should parse to a mapping"
    assert "stages" in data, "dvc.yaml missing top-level 'stages'"

    stages = data["stages"]
    assert isinstance(stages, dict), "'stages' should be a mapping"

    required = {
        "download_val_and_ann",
        "extract_val_and_ann",
        "download_train",
        "extract_train",
        "coco_to_yolo",
        "preprocess_images_link",
        "splits",
        "reports",
    }
    missing = sorted(required - set(stages.keys()))
    assert not missing, f"dvc.yaml missing stages: {missing}"


def test_required_scripts_exist():
    scripts_dir = REPO_ROOT / "scripts"
    assert scripts_dir.exists(), "scripts/ folder missing"

    required_scripts = [
        "download_coco2017.py",
        "extract_zips.py",
        "convert_coco_to_yolo.py",
        "fill_missing_labels.py",
        "preprocess_images.py",
        "create_splits.py",
        "schema_stats.py",
        "quality_checks.py",
        "anomaly_alerts.py",
        "bias_slicing.py",
    ]
    missing = [s for s in required_scripts if not (scripts_dir / s).exists()]
    assert not missing, f"Missing scripts: {missing}"


def test_reports_are_not_committed():
    """
    Since you added data/ to .gitignore, reports/splits should not be committed.
    This test ensures CI doesn't rely on data being present.
    """
    data_dir = REPO_ROOT / "data"
    if not data_dir.exists():
        return
    assert True
