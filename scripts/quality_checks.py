import argparse, json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    return cls, x, y, w, h

def main(processed: str, splits: str, out: str):
    proc = Path(processed)
    sp = Path(splits)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    labels_train = proc / "labels" / "train2017"
    labels_val = proc / "labels" / "val2017"

    # read split lists (full paths)
    train_imgs = (sp / "train.txt").read_text().splitlines()
    val_imgs = (sp / "val.txt").read_text().splitlines()
    test_imgs = (sp / "test.txt").read_text().splitlines()

    report = {
        "total_images": {"train": len(train_imgs), "val": len(val_imgs), "test": len(test_imgs)},
        "missing_label_files": {"train": 0, "val": 0, "test": 0},
        "empty_label_files": {"train": 0, "val": 0, "test": 0},
        "invalid_yolo_lines": 0,
        "invalid_bbox_range": 0,
        "class_id_out_of_range": 0,
        "max_class_id_seen": -1,
        "class_histogram_top20": [],
        "objects_per_image_stats": {},
    }

    # class count + objects per image buckets
    class_counts = Counter()
    obj_counts = {"train": [], "val": [], "test": []}

    def check_split(img_list, split_name):
        nonlocal report
        for img_path_str in tqdm(img_list, desc=f"qc:{split_name}"):
            img_path = Path(img_path_str)
            stem = img_path.stem

            if "train2017" in img_path_str:
                lab = labels_train / f"{stem}.txt"
            elif "val2017" in img_path_str:
                lab = labels_val / f"{stem}.txt"
            else:
                # should not happen
                lab = labels_train / f"{stem}.txt"

            if not lab.exists():
                report["missing_label_files"][split_name] += 1
                obj_counts[split_name].append(0)
                continue

            txt = lab.read_text().strip()
            if txt == "":
                report["empty_label_files"][split_name] += 1
                obj_counts[split_name].append(0)
                continue

            lines = txt.splitlines()
            obj_counts[split_name].append(len(lines))

            for line in lines:
                parsed = parse_yolo_line(line)
                if parsed is None:
                    report["invalid_yolo_lines"] += 1
                    continue
                cls, x, y, w, h = parsed
                report["max_class_id_seen"] = max(report["max_class_id_seen"], cls)
                class_counts[cls] += 1

                # yolo normalized coords must be in [0,1]
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    report["invalid_bbox_range"] += 1
                if cls < 0:
                    report["class_id_out_of_range"] += 1

    check_split(train_imgs, "train")
    check_split(val_imgs, "val")
    check_split(test_imgs, "test")

    # object-per-image stats
    def stats(arr):
        if not arr:
            return {"min": 0, "max": 0, "mean": 0}
        return {
            "min": int(min(arr)),
            "max": int(max(arr)),
            "mean": float(sum(arr) / len(arr)),
        }

    report["objects_per_image_stats"] = {k: stats(v) for k, v in obj_counts.items()}

    # top 20 classes by count (store as strings to avoid huge output)
    top20 = class_counts.most_common(20)
    report["class_histogram_top20"] = [{"class_id": int(k), "count": int(v)} for k, v in top20]

    outp.write_text(json.dumps(report, indent=2))
    print("Wrote quality report:", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.processed, args.splits, args.out)
