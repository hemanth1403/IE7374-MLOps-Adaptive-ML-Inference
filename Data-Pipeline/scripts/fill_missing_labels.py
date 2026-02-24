from __future__ import annotations

import argparse
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_empty_labels_from_raw(raw: Path, processed: Path) -> int:
    """
    Create empty YOLO label files for every image in raw/{train2017,val2017}.
    Labels live in processed/labels/{train2017,val2017}.
    """
    created = 0
    for split in ["train2017", "val2017"]:
        img_dir = raw / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing raw images folder: {img_dir}")

        out_dir = processed / "labels" / split
        ensure_dir(out_dir)

        for img_path in img_dir.glob("*.jpg"):
            label_path = out_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                label_path.write_text("", encoding="utf-8")
                created += 1

    return created


def make_empty_labels_from_processed_images(processed: Path) -> int:
    """
    Fallback: Create empty labels based on processed/images/{train2017,val2017}.
    Useful if images are linked/copied already.
    """
    created = 0
    images_root = processed / "images"
    labels_root = processed / "labels"

    for split in ["train2017", "val2017"]:
        img_dir = images_root / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing processed images folder: {img_dir}")

        out_dir = labels_root / split
        ensure_dir(out_dir)

        for img_path in img_dir.glob("*.jpg"):
            label_path = out_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                label_path.write_text("", encoding="utf-8")
                created += 1

    return created


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--raw", required=False, type=Path, help="Path to data/raw (contains train2017/val2017)")
    args = ap.parse_args()

    processed = args.processed
    ensure_dir(processed / "labels")

    if args.raw is not None:
        created = make_empty_labels_from_raw(args.raw, processed)
    else:
        created = make_empty_labels_from_processed_images(processed)

    print(f"Created {created} empty label files")


if __name__ == "__main__":
    main()
