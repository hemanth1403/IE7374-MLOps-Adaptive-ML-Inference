import argparse, hashlib
from pathlib import Path

def stable_u01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def main(processed: str, out: str, test_frac: float):
    p = Path(processed)
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)

    train_imgs = sorted((p / "images" / "train2017").glob("*.jpg"))
    val_imgs = sorted((p / "images" / "val2017").glob("*.jpg"))

    train_list, test_list = [], []
    for img in train_imgs:
        u = stable_u01(img.name)
        if u < test_frac:
            test_list.append(img)
        else:
            train_list.append(img)

    def write_list(path: Path, imgs):
        path.write_text("\n".join(str(x) for x in imgs) + "\n")

    write_list(outp / "train.txt", train_list)
    write_list(outp / "val.txt", val_imgs)       # official val2017
    write_list(outp / "test.txt", test_list)     # internal holdout from train

    print(f"train={len(train_list)} val={len(val_imgs)} test={len(test_list)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.10)
    args = ap.parse_args()
    main(args.processed, args.out, args.test_frac)
