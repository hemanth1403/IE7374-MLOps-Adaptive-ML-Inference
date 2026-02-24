from pathlib import Path
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]   # Data-Pipeline/
REPO_ROOT = PIPELINE_ROOT.parent                      # repo root


def test_repo_has_expected_structure():
    """CI-safe sanity checks: no data, no DVC, no downloads."""
    assert (PIPELINE_ROOT / "dvc.yaml").exists(), "dvc.yaml missing in Data-Pipeline/"
    assert (PIPELINE_ROOT / "scripts").exists(), "scripts/ folder missing in Data-Pipeline/"
    assert (PIPELINE_ROOT / "requirements.txt").exists(), "requirements.txt missing in Data-Pipeline/"
    assert (REPO_ROOT / ".github" / "workflows").exists(), ".github/workflows missing (CI not set up?)"


def test_dvc_yaml_parses_and_has_required_stages():
    dvc_path = PIPELINE_ROOT / "dvc.yaml"
    doc = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))
    assert "stages" in doc, "dvc.yaml must define 'stages'"
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
    missing = required - set(doc["stages"].keys())
    assert not missing, f"Missing stages: {missing}"