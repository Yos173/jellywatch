"""
Run the jellyfish detector on an image, a video file or a webcam.

Outputs
  * annotated image / video with boxes + a heads-up display (count, density, level)
  * CSV with one row per processed frame:
        timestamp, video_time_s, frame, count, density, unique_tracked
  * printed summary

Definitions (simple enough to explain to judges)
  count          = jellyfish detected in this frame
  density        = rolling mean of `count` over the last N frames (smooths flicker)
  unique_tracked = distinct jellyfish IDs seen so far (ByteTrack tracker), i.e.
                   how many individual animals passed the camera

Usage:
    python src/detection/detect.py --source data/videos/demo_bloom.mp4
    python src/detection/detect.py --source some_image.jpg
    python src/detection/detect.py --source 0 --show          # webcam
"""
import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (DENSITY_BLOOM_THRESHOLDS, DENSITY_WINDOW, DETECT_CONF,  # noqa: E402
                        DETECTOR_WEIGHTS, OUTPUTS_DIR, level_color)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOX_COLOR = (255, 220, 0)        # BGR cyan-ish
TEXT_COLOR = (255, 255, 255)


def density_level(density: float) -> str:
    """Translate live camera density into an alert level."""
    level = "Low"
    for name, threshold in DENSITY_BLOOM_THRESHOLDS.items():   # ordered Medium -> Critical
        if density >= threshold:
            level = name
    return level


def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class JellyDetector:
    """Wraps a YOLO model and keeps the running statistics for one stream."""

    def __init__(self, weights: Path = DETECTOR_WEIGHTS, conf: float = DETECT_CONF,
                 track: bool = True, window: int = DENSITY_WINDOW, imgsz: int = 640):
        from ultralytics import YOLO
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Detector weights not found at {weights}. Train first: python src/detection/train.py")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.track = track
        self.imgsz = imgsz
        self.history = deque(maxlen=window)
        self.seen_ids = set()

    def reset(self) -> None:
        self.history.clear()
        self.seen_ids.clear()

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Detect jellyfish in one BGR frame. Returns (annotated_frame, stats)."""
        if self.track:
            res = self.model.track(frame, conf=self.conf, imgsz=self.imgsz, persist=True,
                                   tracker="bytetrack.yaml", verbose=False)[0]
        else:
            res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int) if len(boxes) else np.empty((0, 4), int)
        confs = boxes.conf.cpu().numpy() if len(boxes) else np.empty(0)
        ids = boxes.id.cpu().numpy().astype(int) if (len(boxes) and boxes.id is not None) else [None] * len(xyxy)
        self.seen_ids.update(i for i in ids if i is not None)

        count = len(xyxy)
        self.history.append(count)
        density = float(np.mean(self.history))
        stats = {
            "count": count,
            "density": round(density, 2),
            "unique_tracked": len(self.seen_ids),
            "level": density_level(density),
            "mean_conf": round(float(confs.mean()), 3) if count else 0.0,
        }
        return self._annotate(frame.copy(), xyxy, confs, ids, stats), stats

    # ------------------------------------------------------------------ drawing
    def _annotate(self, img, xyxy, confs, ids, stats) -> np.ndarray:
        h, w = img.shape[:2]
        scale = max(w, h) / 1280
        thick = max(2, int(2 * scale))
        for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thick)
            label = f"jellyfish {c:.2f}" + (f" #{tid}" if tid is not None else "")
            fs = 0.5 * max(scale, 0.6)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), BOX_COLOR, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 30, 0), 1, cv2.LINE_AA)

        # Heads-up display panel (semi-transparent)
        pw, ph = int(330 * max(scale, 0.6)), int(130 * max(scale, 0.6))
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + pw, 10 + ph), (40, 25, 10), -1)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
        fs = 0.6 * max(scale, 0.6)
        lh = int(26 * max(scale, 0.6))
        y = 10 + lh
        cv2.putText(img, "JellyWatch AI", (22, y), cv2.FONT_HERSHEY_DUPLEX, fs * 1.1, (255, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(img, f"Jellyfish in frame: {stats['count']}", (22, y + lh), cv2.FONT_HERSHEY_SIMPLEX, fs, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"Density (smoothed): {stats['density']:.1f}", (22, y + 2 * lh), cv2.FONT_HERSHEY_SIMPLEX, fs, TEXT_COLOR, 1, cv2.LINE_AA)
        lvl = stats["level"]
        cv2.putText(img, f"Unique tracked: {stats['unique_tracked']}   Level: {lvl}", (22, y + 3 * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, _hex_to_bgr(level_color(lvl)), 1, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------- sources
def open_source(source: str):
    """Return (kind, cv2.VideoCapture | image, fps)."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        return "webcam", cap, cap.get(cv2.CAP_PROP_FPS) or 25
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(source)
    if path.suffix.lower() in IMAGE_EXTS:
        return "image", cv2.imread(str(path)), 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {source}")
    return "video", cap, cap.get(cv2.CAP_PROP_FPS) or 25


def stream(source: str, detector: JellyDetector, stride: int = 1, max_frames: int | None = None,
           resize_width: int | None = 1280):
    """Generator used by both the CLI and the dashboard.
    Yields (annotated_frame_bgr, row_dict) for each processed frame."""
    kind, src, fps = open_source(source)
    start = datetime.now()
    if kind == "image":
        annotated, stats = detector.process(src)
        yield annotated, {"timestamp": start.isoformat(timespec="seconds"), "video_time_s": 0.0, "frame": 0, **stats}
        return

    idx = processed = 0
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            if idx % stride == 0:
                if resize_width and frame.shape[1] > resize_width:
                    f = resize_width / frame.shape[1]
                    frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * f)))
                annotated, stats = detector.process(frame)
                t = idx / fps
                ts = datetime.now() if kind == "webcam" else start + timedelta(seconds=t)
                yield annotated, {"timestamp": ts.isoformat(timespec="milliseconds"),
                                  "video_time_s": round(t, 2), "frame": idx, **stats}
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            idx += 1
    finally:
        src.release()


def _video_writer(path: Path, fps: float, size: tuple) -> cv2.VideoWriter:
    # 'mp4v' ships with opencv-python everywhere. (H.264/'avc1' needs an extra
    # OpenH264 DLL on Windows and can report isOpened()=True while silently failing.)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not wr.isOpened():
        raise RuntimeError("OpenCV could not create an MP4 writer")
    return wr


def run(source: str, out_dir: Path = OUTPUTS_DIR, show: bool = False, stride: int = 1,
        max_frames: int | None = None, conf: float = DETECT_CONF, track: bool = True,
        weights: Path = DETECTOR_WEIGHTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = JellyDetector(weights=weights, conf=conf, track=track)
    kind, _, fps = open_source(source)
    stem = "webcam" if source.isdigit() else Path(source).stem
    csv_path = out_dir / f"{stem}_detections.csv"
    media_path = out_dir / (f"{stem}_annotated.jpg" if kind == "image" else f"{stem}_annotated.mp4")

    writer, rows, t0 = None, [], time.time()
    for annotated, row in stream(source, detector, stride=stride, max_frames=max_frames):
        rows.append(row)
        if kind == "image":
            cv2.imwrite(str(media_path), annotated)
        else:
            if writer is None:
                writer = _video_writer(media_path, max(fps / stride, 1), (annotated.shape[1], annotated.shape[0]))
            writer.write(annotated)
        if show:
            cv2.imshow("JellyWatch AI", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if len(rows) % 50 == 0:
            print(f"  frame {row['frame']:>5}: count={row['count']:>2} density={row['density']:.2f} level={row['level']}")
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    fields = ["timestamp", "video_time_s", "frame", "count", "density", "unique_tracked", "level", "mean_conf"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = [r["count"] for r in rows] or [0]
    summary = {
        "source": source,
        "frames_processed": len(rows),
        "max_count": int(max(counts)),
        "mean_count": round(float(np.mean(counts)), 2),
        "peak_density": round(max((r["density"] for r in rows), default=0), 2),
        "unique_tracked": rows[-1]["unique_tracked"] if rows else 0,
        "final_level": rows[-1]["level"] if rows else "Low",
        "fps_processing": round(len(rows) / max(time.time() - t0, 1e-6), 1),
        "annotated": str(media_path),
        "csv": str(csv_path),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="JellyWatch AI — jellyfish detection")
    ap.add_argument("--source", required=True, help="image, video file or webcam index (0)")
    ap.add_argument("--out", default=str(OUTPUTS_DIR), help="output folder")
    ap.add_argument("--weights", default=str(DETECTOR_WEIGHTS), help="YOLO weights (.pt)")
    ap.add_argument("--conf", type=float, default=DETECT_CONF)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (speed-up on CPU)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-track", action="store_true", help="disable ByteTrack IDs")
    ap.add_argument("--show", action="store_true", help="show a live window (press q to quit)")
    args = ap.parse_args()

    s = run(args.source, Path(args.out), show=args.show, stride=args.stride,
            max_frames=args.max_frames, conf=args.conf, track=not args.no_track, weights=Path(args.weights))
    print("\n=== JellyWatch detection summary ===")
    for k, v in s.items():
        print(f"{k:>17}: {v}")


if __name__ == "__main__":
    main()
