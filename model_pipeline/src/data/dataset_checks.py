from __future__ import annotations

from pathlib import Path
import json
import sys
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


def main() -> int:
    config_path = REPO_ROOT / "model_pipeline" / "configs" / "data" / "dataset_config.yaml"
    cfg = load_yaml(config_path)

    required_paths = {
        key: resolve_repo_path(value)
        for key, value in cfg["paths"].items()
    }

    missing = {}
    for key, path in required_paths.items():
        if not path.exists():
            missing[key] = str(path)

    print("Dataset path check:")
    for key, path in required_paths.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"- {key}: {status} -> {path}")

    if missing:
        print("\nMissing required dataset inputs:")
        print(json.dumps(missing, indent=2))
        return 1

    print("\nAll required dataset inputs are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())