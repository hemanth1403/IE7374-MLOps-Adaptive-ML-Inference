import argparse, json
from pathlib import Path

def main(processed: str, splits: str, out_dir: str):
    proc = Path(processed)
    sp = Path(splits)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    schema = {
        "images": {
            "train_dir": "data/processed/images/train2017",
            "val_dir": "data/processed/images/val2017",
            "format": "jpg",
        },
        "labels": {
            "format": "yolo",
            "train_dir": "data/processed/labels/train2017",
            "val_dir": "data/processed/labels/val2017",
            "line_format": "class_id x_center y_center width height (normalized)",
        },
        "splits": {
            "train_txt": "data/splits/train.txt",
            "val_txt": "data/splits/val.txt",
            "test_txt": "data/splits/test.txt",
        }
    }

    stats = {
        "num_images": {
            "train": len((sp / "train.txt").read_text().splitlines()),
            "val": len((sp / "val.txt").read_text().splitlines()),
            "test": len((sp / "test.txt").read_text().splitlines()),
        },
        "num_label_files": {
            "train": len(list((proc / "labels" / "train2017").glob("*.txt"))),
            "val": len(list((proc / "labels" / "val2017").glob("*.txt"))),
        },
        "num_classes": len((proc / "coco.names").read_text().splitlines())
    }

    (outp / "schema.json").write_text(json.dumps(schema, indent=2))
    (outp / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Wrote schema.json and stats.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.processed, args.splits, args.out)
