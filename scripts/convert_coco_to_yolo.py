import argparse, json
from pathlib import Path
from tqdm import tqdm

def coco_to_yolo_bbox(bbox_xywh, img_w, img_h):
    x, y, w, h = bbox_xywh
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return xc, yc, wn, hn

def write_labels(coco_json: Path, out_labels_dir: Path, out_names: Path):
    data = json.loads(coco_json.read_text())

    # category_id -> contiguous class index
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    out_names.write_text("\n".join(names))

    img_info = {img["id"]: img for img in data["images"]}

    ann_by_img = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_id, anns in tqdm(ann_by_img.items(), desc=f"labels:{coco_json.name}"):
        img = img_info.get(image_id)
        if not img:
            continue
        file_name = img["file_name"]
        w, h = img["width"], img["height"]

        label_path = out_labels_dir / (Path(file_name).stem + ".txt")
        lines = []
        for ann in anns:
            cls = cat_id_to_idx[ann["category_id"]]
            xc, yc, wn, hn = coco_to_yolo_bbox(ann["bbox"], w, h)

            # clamp to [0,1]
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

def main(raw: str, out: str):
    rawp = Path(raw)
    outp = Path(out)

    ann_dir = rawp / "annotations"
    train_json = ann_dir / "instances_train2017.json"
    val_json = ann_dir / "instances_val2017.json"

    (outp / "labels" / "train2017").mkdir(parents=True, exist_ok=True)
    (outp / "labels" / "val2017").mkdir(parents=True, exist_ok=True)

    names_path = outp / "coco.names"
    write_labels(train_json, outp / "labels" / "train2017", names_path)
    write_labels(val_json, outp / "labels" / "val2017", names_path)

    print("Wrote YOLO labels + coco.names")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.raw, args.out)
