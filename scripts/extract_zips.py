import argparse
from pathlib import Path
import zipfile

def unzip(zip_path: Path, out_dir: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing: {zip_path}")
    print(f"Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main(zips_dir: str, raw_dir: str, subset: str):
    zdir = Path(zips_dir)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    if subset in ("val", "all"):
        unzip(zdir / "val2017.zip", raw)
        unzip(zdir / "annotations_trainval2017.zip", raw)

    if subset in ("train", "all"):
        unzip(zdir / "train2017.zip", raw)

    print("Done extracting. Expect raw/val2017, raw/train2017, raw/annotations")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--subset", choices=["val", "train", "all"], default="val")
    args = ap.parse_args()
    main(args.zips, args.raw, args.subset)
