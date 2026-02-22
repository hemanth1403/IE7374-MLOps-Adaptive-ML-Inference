import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

def count_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    n = 0
    # fast line counting; avoids read_text() on huge IO
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def bucket(n: int) -> str:
    if n <= 2:
        return "simple (<=2 objects)"
    if n <= 7:
        return "moderate (3-7 objects)"
    return "complex (>=8 objects)"

def main(processed: str, splits: str, out: str):
    proc = Path(processed)
    sp = Path(splits)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    labels_train = proc / "labels" / "train2017"
    labels_val = proc / "labels" / "val2017"

    def load_list(p: Path):
        return p.read_text().splitlines()

    split_lists = {
        "train": load_list(sp / "train.txt"),
        "val": load_list(sp / "val.txt"),
        "test": load_list(sp / "test.txt"),
    }

    counts = defaultdict(Counter)

    for split_name in ["train", "val", "test"]:
        imgs = split_lists[split_name]
        for img_path_str in tqdm(imgs, desc=f"slice:{split_name}", unit="img"):
            img_path = Path(img_path_str)
            stem = img_path.stem
            if "train2017" in img_path_str:
                lab = labels_train / f"{stem}.txt"
            else:
                lab = labels_val / f"{stem}.txt"

            n = count_objects(lab)
            counts[split_name][bucket(n)] += 1

    lines = ["# Bias / Slicing Report", "", "## Slice by scene complexity (object count tiers)", ""]
    for split_name in ["train", "val", "test"]:
        lines.append(f"### {split_name}")
        total = sum(counts[split_name].values()) or 1
        for k in ["simple (<=2 objects)", "moderate (3-7 objects)", "complex (>=8 objects)"]:
            v = counts[split_name].get(k, 0)
            pct = 100.0 * v / total
            lines.append(f"- {k}: {v} ({pct:.2f}%)")
        lines.append("")

    outp.write_text("\n".join(lines))
    print("Wrote bias slicing report:", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.processed, args.splits, args.out)
