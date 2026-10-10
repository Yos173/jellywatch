"""
Download public jellyfish datasets and merge them into one 1-class YOLO dataset.
No account or API key needed (both are mirrored on Hugging Face, CC BY 4.0).

1. "Aquarium" (Roboflow, LibreYOLO/aquarium-qlnqy) — aquarium photos, 7 classes.
2. "Brackish" (Aalborg University, CVPR-W 2019, moondream/brackish_underwater) —
   frames from a PERMANENTLY MOUNTED underwater camera in a sea strait. This is
   exactly the "fixed camera at a cooling-water intake" scenario.
3. Open Images V7, class "Jellyfish" (Google; boxes CC BY 4.0, images CC BY 2.0) —
   ~870 diverse close-up photos. Downloaded from the public S3 bucket.

We keep ONLY jellyfish as class 0. Images of other animals stay in the dataset as
"negatives", which teaches the model NOT to fire on fish/crabs/sharks/etc.

Usage:
    python src/detection/prepare_dataset.py
"""
import io
import random
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import IMAGES_DIR, DATASET_YAML  # noqa: E402

HF_REPO = "LibreYOLO/aquarium-qlnqy"
RAW_DIR = IMAGES_DIR / "aquarium_raw"
BRACKISH_REPO = "moondream/brackish_underwater"
BRACKISH_DIR = IMAGES_DIR / "brackish_parquet"
BRACKISH_NEG_PER_POS = 1.0      # negatives (no jellyfish) kept per positive image
OUT_DIR = DATASET_YAML.parent
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def download() -> Path:
    print(f"Downloading {HF_REPO} from Hugging Face ...")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=RAW_DIR)
    return RAW_DIR


def convert_aquarium(raw: Path) -> dict:
    names = yaml.safe_load((raw / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    jelly_id = next(i for i, n in enumerate(names) if "jelly" in n.lower())
    print(f"Source classes: {names} -> keeping '{names[jelly_id]}' (id {jelly_id}) as class 0")

    stats = {}
    for src_split, dst_split in SPLITS.items():
        src_img = raw / src_split / "images"
        if not src_img.exists():
            continue
        dst_img = OUT_DIR / dst_split / "images"
        dst_lbl = OUT_DIR / dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_img = n_box = n_pos = 0
        for img in sorted(src_img.glob("*.jpg")):
            shutil.copy2(img, dst_img / img.name)
            src_lbl = raw / src_split / "labels" / (img.stem + ".txt")
            lines = []
            if src_lbl.exists():
                for line in src_lbl.read_text().splitlines():
                    parts = line.split()
                    if parts and int(parts[0]) == jelly_id:
                        lines.append("0 " + " ".join(parts[1:]))
            (dst_lbl / (img.stem + ".txt")).write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
            n_pos += bool(lines)
        stats[dst_split] = (n_img, n_pos, n_box)
    return stats


def convert_brackish(seed: int = 0) -> dict:
    """Extract jellyfish frames (+ an equal number of negatives) from Brackish parquet files."""
    print(f"Downloading {BRACKISH_REPO} from Hugging Face (~540 MB) ...")
    snapshot_download(repo_id=BRACKISH_REPO, repo_type="dataset", local_dir=BRACKISH_DIR,
                      allow_patterns=["data/*.parquet"])
    rng = random.Random(seed)
    stats = {}
    for src_split, dst_split in SPLITS.items():
        files = list((BRACKISH_DIR / "data").glob(f"{src_split}-*.parquet"))
        if not files:
            continue
        pf = pq.ParquetFile(files[0])
        objects = pq.read_table(files[0], columns=["objects"]).column("objects").to_pylist()
        jelly = [[b for o in objs if o["name"] == "jellyfish" for b in o["boxes"]] for objs in objects]
        pos = [i for i, b in enumerate(jelly) if b]
        neg = [i for i, b in enumerate(jelly) if not b]
        keep = set(pos) | set(rng.sample(neg, min(len(neg), int(len(pos) * BRACKISH_NEG_PER_POS))))

        dst_img = OUT_DIR / dst_split / "images"
        dst_lbl = OUT_DIR / dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        idx = n_box = 0
        for batch in pf.iter_batches(batch_size=256, columns=["image"]):
            for img in batch.column("image").to_pylist():
                if idx in keep:
                    name = f"brackish_{src_split}_{idx:05d}"
                    im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                    im.thumbnail((1280, 1280))            # 1920x1080 -> 1280x720 (saves disk)
                    im.save(dst_img / f"{name}.jpg", quality=92)
                    # boxes are [x_min, y_min, x_max, y_max] normalised -> YOLO cx cy w h
                    lines = [f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                             for x1, y1, x2, y2 in jelly[idx]]
                    (dst_lbl / f"{name}.txt").write_text("\n".join(lines))
                    n_box += len(lines)
                idx += 1
        stats[dst_split] = (len(keep), len(pos), n_box)
    return stats


OI_DIR = IMAGES_DIR / "openimages"
OI_CLASS = "/m/0d8zb"           # Open Images label id for "Jellyfish"
OI_CSV = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",   # 2.2 GB, streamed
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
OI_SPLITS = {"train": "train", "validation": "val", "test": "test"}


def convert_openimages() -> dict:
    """Open Images V7 'Jellyfish' boxes (annotations CC BY 4.0, images CC BY 2.0).
    Diverse close-up photos — closes the gap between small far-away jellyfish (Brackish)
    and large translucent ones near a camera."""
    from concurrent.futures import ThreadPoolExecutor

    import requests

    OI_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, dst_split in OI_SPLITS.items():
        # 1) keep only jellyfish rows (the train CSV is streamed, never stored)
        rows_file = OI_DIR / f"{split}_jelly.csv"
        if not rows_file.exists():
            print(f"Filtering Open Images {split} annotations ...")
            with requests.get(OI_CSV[split], stream=True, timeout=120) as r, open(rows_file, "w") as out:
                for line in r.iter_lines(decode_unicode=True):
                    if line and f",{OI_CLASS}," in line:
                        out.write(line + "\n")
        boxes: dict[str, list] = {}
        for line in rows_file.read_text().splitlines():
            p = line.split(",")
            img_id, x1, x2, y1, y2 = p[0], *map(float, p[4:8])
            boxes.setdefault(img_id, []).append((x1, y1, x2, y2))

        # 2) download the images in parallel and write YOLO labels
        dst_img = OUT_DIR / dst_split / "images"
        dst_lbl = OUT_DIR / dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        cache = OI_DIR / "images"
        cache.mkdir(exist_ok=True)

        def fetch(img_id: str) -> bool:
            src = cache / f"{img_id}.jpg"
            if not src.exists():
                for _ in range(3):                      # S3 occasionally times out -> retry
                    try:
                        r = requests.get(f"https://s3.amazonaws.com/open-images-dataset/{split}/{img_id}.jpg",
                                         timeout=60)
                        if not r.ok:
                            return False
                        im = Image.open(io.BytesIO(r.content)).convert("RGB")
                        break
                    except Exception:
                        continue
                else:
                    return False                        # skip this image after 3 failures
                im.thumbnail((1024, 1024))
                im.save(src, quality=92)
            shutil.copy2(src, dst_img / f"oi_{img_id}.jpg")
            lines = [f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                     for x1, y1, x2, y2 in boxes[img_id]]
            (dst_lbl / f"oi_{img_id}.txt").write_text("\n".join(lines))
            return True

        with ThreadPoolExecutor(16) as pool:
            ok = [i for i, good in zip(boxes, pool.map(fetch, boxes)) if good]
        stats[dst_split] = (len(ok), len(ok), sum(len(boxes[i]) for i in ok))
    return stats


def write_yaml() -> None:
    DATASET_YAML.write_text(yaml.safe_dump({
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "jellyfish"},
    }, sort_keys=False))


def print_stats(title: str, stats: dict) -> None:
    print(f"\n{title}\nSplit   images  with-jellyfish  jellyfish-boxes")
    for split, (n_img, n_pos, n_box) in stats.items():
        print(f"{split:<7} {n_img:>6}  {n_pos:>14}  {n_box:>15}")


if __name__ == "__main__":
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)          # rebuild from scratch (raw downloads are cached)
    print_stats("Aquarium (Roboflow)", convert_aquarium(download()))
    print_stats("Brackish (Aalborg Univ., fixed underwater camera)", convert_brackish())
    print_stats("Open Images V7 — Jellyfish (Google)", convert_openimages())
    write_yaml()
    print(f"\nDataset ready: {DATASET_YAML}")
