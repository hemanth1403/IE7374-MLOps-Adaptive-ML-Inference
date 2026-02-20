import argparse
from pathlib import Path
import requests
from tqdm import tqdm

URLS = {
    "train": {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    },
    "val": {
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    },
    "all": {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
}

def _remote_size(url: str) -> int | None:
    # Try HEAD first to get Content-Length
    try:
        h = requests.head(url, timeout=30, allow_redirects=True)
        if h.status_code >= 400:
            return None
        cl = h.headers.get("content-length")
        return int(cl) if cl is not None else None
    except Exception:
        return None

def download_resume(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = out_path.stat().st_size if out_path.exists() else 0

    remote = _remote_size(url)
    if remote is not None and existing == remote and remote > 0:
        print(f"[skip] already complete: {out_path.name} ({existing} bytes)")
        return
    if remote is not None and existing > remote and remote > 0:
        print(f"[warn] local file bigger than remote, re-downloading: {out_path.name}")
        out_path.unlink(missing_ok=True)
        existing = 0

    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()

        remaining = r.headers.get("content-length")
        total = (int(remaining) + existing) if remaining is not None else None
        mode = "ab" if existing > 0 else "wb"
        desc = f"{out_path.name} (resume)" if existing > 0 else out_path.name

        with open(out_path, mode) as f, tqdm(
            total=total, initial=existing, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main(out_dir: str, subset: str):
    out = Path(out_dir)
    for fname, url in URLS[subset].items():
        print(f"Downloading: {fname}")
        download_resume(url, out / fname)
    print("Done downloading.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=["val", "train", "all"], default="val")
    args = ap.parse_args()
    main(args.out, args.subset)
