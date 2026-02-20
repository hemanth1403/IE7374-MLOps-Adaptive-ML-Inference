import argparse, os, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)  # works in many setups; otherwise falls back
    except Exception:
        shutil.copy2(src, dst)

def resize(src: Path, dst: Path, size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    img = img.resize((size, size))
    img.save(dst, quality=90)

def process_dir(src_dir: Path, dst_dir: Path, mode: str, size: int):
    ensure_dir(dst_dir)
    imgs = list(src_dir.glob("*.jpg"))
    for p in tqdm(imgs, desc=f"{mode}:{src_dir.name}"):
        out = dst_dir / p.name
        if mode == "resize":
            resize(p, out, size)
        else:
            link_or_copy(p, out)

def main(raw: str, processed: str, mode: str, size: int):
    rawp = Path(raw)
    outp = Path(processed)

    process_dir(rawp / "train2017", outp / "images" / "train2017", mode, size)
    process_dir(rawp / "val2017", outp / "images" / "val2017", mode, size)

    print("Done preprocessing images.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--mode", choices=["link", "resize"], default="link")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    main(args.raw, args.processed, args.mode, args.size)
