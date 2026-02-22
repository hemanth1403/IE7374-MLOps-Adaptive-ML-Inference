import argparse, json
from pathlib import Path

def main(quality: str):
    q = json.loads(Path(quality).read_text())

    # thresholds (reasonable defaults)
    max_missing = 50        # should be near-zero
    max_invalid_bbox = 0    # should be zero
    max_invalid_lines = 0   # should be zero

    missing_total = sum(q["missing_label_files"].values())

    problems = []
    if missing_total > max_missing:
        problems.append(f"missing_label_files_total={missing_total} > {max_missing}")
    if q["invalid_bbox_range"] > max_invalid_bbox:
        problems.append(f"invalid_bbox_range={q['invalid_bbox_range']} > {max_invalid_bbox}")
    if q["invalid_yolo_lines"] > max_invalid_lines:
        problems.append(f"invalid_yolo_lines={q['invalid_yolo_lines']} > {max_invalid_lines}")

    if problems:
        raise SystemExit("ALERT: " + "; ".join(problems))

    print("No anomalies detected.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", required=True)
    args = ap.parse_args()
    main(args.quality)
