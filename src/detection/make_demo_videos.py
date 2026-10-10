"""
Download free jellyfish footage from Wikimedia Commons and cut short MP4 demo clips.

  demo_bloom.mp4    <- "Oorkwal in een school van duizenden" (moon jellyfish bloom,
                       Netherlands; CC BY-SA 3.0 NL, Wikimedia Commons). Murky green
                       water with hundreds of jellyfish — realistic bloom conditions.
  demo_aquarium.mp4 <- "Moon Jellies Swimming" (CC BY-SA 2.0, Wikimedia Commons).

Usage:
    python src/detection/make_demo_videos.py
"""
import sys
from pathlib import Path

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import VIDEOS_DIR  # noqa: E402

UA = {"User-Agent": "JellyWatchAI/1.0 (hackathon demo; educational use)"}
CLIPS = [
    {
        "out": "demo_bloom.mp4",
        "raw": "bloom_480p.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/94/"
               "Oorkwal_in_een_school_van_duizenden-4931717.webm/"
               "Oorkwal_in_een_school_van_duizenden-4931717.webm.480p.vp9.webm",
        "start_s": 24, "duration_s": 40,
    },
    {
        "out": "demo_aquarium.mp4",
        "raw": "Moon_Jellies_Swimming.ogv",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a6/Moon_Jellies_Swimming.ogv",
        "start_s": 0, "duration_s": 22,
    },
]


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 100_000:
        return
    print(f"Downloading {path.name} ...")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def cut(src: Path, dst: Path, start_s: float, duration_s: float, width: int = 960) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    writer = None
    for _ in range(int(duration_s * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        h = int(frame.shape[0] * width / frame.shape[1]) // 2 * 2
        frame = cv2.resize(frame, (width, h))
        if writer is None:
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, h))
        writer.write(frame)
    cap.release()
    if writer:
        writer.release()
    print(f"  -> {dst} ({duration_s}s)")


if __name__ == "__main__":
    raw_dir = VIDEOS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for c in CLIPS:
        raw = raw_dir / c["raw"]
        download(c["url"], raw)
        cut(raw, VIDEOS_DIR / c["out"], c["start_s"], c["duration_s"])
