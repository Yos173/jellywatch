"""
Fine-tune a small YOLO model to detect jellyfish.

Starts from COCO-pretrained YOLO11s weights (transfer learning) and trains on
the 1-class jellyfish dataset built by prepare_dataset.py.
GPU (RTX 2060, plugged in): ~20-25 min. Laptop on battery: ~70 min.
CPU only / Colab-free: use --epochs 20 --imgsz 512 (~1 h).

Usage:
    python src/detection/train.py                 # defaults
    python src/detection/train.py --epochs 25 --device cpu
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import BASE_YOLO_MODEL, DATASET_YAML, DETECTOR_WEIGHTS, MODELS_DIR, ROOT  # noqa: E402


def main() -> None:
    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description="Fine-tune YOLO on jellyfish")
    ap.add_argument("--model", default=BASE_YOLO_MODEL, help="base weights (yolov8n.pt, yolo11s.pt, ...)")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not DATASET_YAML.exists():
        sys.exit("Dataset not found. Run: python src/detection/prepare_dataset.py")

    print(f"Training {args.model} on {DATASET_YAML} | device={args.device} epochs={args.epochs}")
    model = YOLO(args.model)
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=25,              # early stop if val mAP stops improving
        project=str(ROOT / "runs"),
        name="jellyfish",
        exist_ok=True,
        workers=4,
        cache="ram",              # decode JPEGs once -> GPU is no longer starved by the CPU
        # Underwater-friendly augmentation: colour shifts + flips, mosaic for small objects
        hsv_h=0.03, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, flipud=0.2, mosaic=1.0, mixup=0.1,
        plots=True,
        verbose=False,
    )

    # Copy best weights to models/best.pt
    best = Path(model.trainer.best)
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, DETECTOR_WEIGHTS)

    # Final evaluation on the held-out TEST split (never seen in training)
    best_model = YOLO(str(DETECTOR_WEIGHTS))
    metrics = {}
    for split in ("val", "test"):
        m = best_model.val(data=str(DATASET_YAML), split=split, imgsz=args.imgsz,
                           device=args.device, verbose=False, plots=False)
        metrics[split] = {
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
        }
    metrics["base_model"] = args.model
    metrics["epochs"] = args.epochs
    (MODELS_DIR / "detector_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Jellyfish detector results ===")
    for split in ("val", "test"):
        r = metrics[split]
        print(f"{split:<5} mAP50={r['mAP50']:.3f}  mAP50-95={r['mAP50_95']:.3f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}")
    print(f"Saved weights -> {DETECTOR_WEIGHTS}")


if __name__ == "__main__":
    main()
