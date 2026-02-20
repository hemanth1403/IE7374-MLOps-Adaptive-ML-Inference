import argparse
from pathlib import Path
from tqdm import tqdm

def ensure_empty_label(label_path: Path):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_path.exists():
        label_path.write_text("")  # empty file = no objects

def main(processed: str):
    proc = Path(processed)

    img_dirs = {
        "train2017": proc / "images" / "train2017",
        "val2017": proc / "images" / "val2017",
    }
    lab_dirs = {
        "train2017": proc / "labels" / "train2017",
        "val2017": proc / "labels" / "val2017",
    }

    total_created = 0
    for split in ["train2017", "val2017"]:
        imgs = sorted(img_dirs[split].glob("*.jpg"))
        for img in tqdm(imgs, desc=f"fill_missing:{split}"):
            lab = lab_dirs[split] / (img.stem + ".txt")
            if not lab.exists():
                ensure_empty_label(lab)
                total_created += 1

    print(f"Created {total_created} empty label files.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    args = ap.parse_args()
    main(args.processed)
