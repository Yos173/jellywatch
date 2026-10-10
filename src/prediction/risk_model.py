"""
Bloom risk model: REAL ocean/weather features -> risk score 0–100 + level.

Train:    python src/prediction/risk_model.py --train
Predict:  python src/prediction/risk_model.py --predict      (trains first if needed)
Options:  --refresh   re-download data from Open-Meteo
          --density 6 blend in a live camera density (jellyfish/frame) at Barakah

Model: scikit-learn GradientBoostingClassifier trained on ~4 years of daily data
for 3 locations, with SYNTHETIC bloom labels (see features.py). Evaluated on the
most recent 20% of days (time-based split, no leakage from the future).
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (DENSITY_BLOOM_THRESHOLDS, ENV_DIR, FORECAST_DAYS, LOCATIONS,  # noqa: E402
                        MODELS_DIR, OUTPUTS_DIR, RISK_MODEL_PATH, risk_level)
from src.prediction.features import (FEATURE_LABELS, FEATURES, build_features,  # noqa: E402
                                     explain_drivers, sst_climatology, synthetic_bloom_labels)
from src.prediction.fetch_data import fetch_all  # noqa: E402

FORECAST_CSV = OUTPUTS_DIR / "forecast.csv"


def _load(key: str, kind: str) -> pd.DataFrame:
    path = ENV_DIR / f"{key}_{kind}.csv"
    stale = kind == "forecast" and path.exists() and \
        pd.read_csv(path, usecols=["date"])["date"].max() < (date.today() + timedelta(days=FORECAST_DAYS)).isoformat()
    if not path.exists() or stale:     # forecast is refreshed automatically once a day
        fetch_all(history=(kind == "history"), forecast=(kind == "forecast"))
    return pd.read_csv(path)


def build_training_set() -> pd.DataFrame:
    frames = []
    for i, (key, loc) in enumerate(LOCATIONS.items()):
        hist = _load(key, "history")
        d = build_features(hist, loc["onshore_from_deg"])
        d["bloom"] = synthetic_bloom_labels(d, seed=42 + i)   # SYNTHETIC labels
        d["location"] = key
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def train() -> dict:
    data = build_training_set().sort_values("date")
    cutoff = data["date"].quantile(0.8)                      # time-based split
    tr, te = data[data["date"] <= cutoff], data[data["date"] > cutoff]

    model = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=0)
    model.fit(tr[FEATURES], tr["bloom"])
    p = model.predict_proba(te[FEATURES])[:, 1]

    importance = (pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
                  .assign(label=lambda x: x["feature"].map(FEATURE_LABELS))
                  .sort_values("importance", ascending=False))
    metrics = {
        "train_days": int(len(tr)), "test_days": int(len(te)),
        "test_period": f"{te['date'].min().date()} .. {te['date'].max().date()}",
        "roc_auc": round(float(roc_auc_score(te["bloom"], p)), 3),
        "brier": round(float(brier_score_loss(te["bloom"], p)), 3),
        "bloom_rate": round(float(data["bloom"].mean()), 3),
        "data_source": sorted(data["source"].unique().tolist()),
        "labels": "SYNTHETIC (expert rule + noise) — see features.py",
        "trained_on": str(date.today()),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, RISK_MODEL_PATH)
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    (MODELS_DIR / "risk_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("=== Bloom risk model trained ===")
    for k, v in metrics.items():
        print(f"{k:>12}: {v}")
    print("\nFeature importance:")
    for _, r in importance.iterrows():
        print(f"  {r['label']:<24} {'#' * int(r['importance'] * 60):<30} {r['importance']:.3f}")
    return metrics


def load_model() -> dict:
    if not RISK_MODEL_PATH.exists():
        train()
    return joblib.load(RISK_MODEL_PATH)


def combine_with_detection(risk: float, density: float | None) -> float:
    """Blend forecast risk with what a camera actually sees right now.
    A camera density at the 'Critical' threshold pushes risk 70% of the way to 100."""
    if density is None or density <= 0:
        return risk
    camera = min(density / DENSITY_BLOOM_THRESHOLDS["Critical"], 1.0)
    return float(risk + (100 - risk) * 0.7 * camera)


def predict_all(refresh: bool = False, density_by_location: dict | None = None,
                days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Risk for today + next `days` days at every location."""
    if refresh:
        fetch_all(history=False, forecast=True)
    model = load_model()["model"]
    rows = []
    for key, loc in LOCATIONS.items():
        hist = _load(key, "history")
        fc = _load(key, "forecast")
        d = build_features(fc, loc["onshore_from_deg"], sst_climatology(hist))
        future = d[d["date"] >= pd.Timestamp(date.today())].head(days + 1)
        probs = model.predict_proba(future[FEATURES])[:, 1]
        density = (density_by_location or {}).get(key)
        for (_, r), p in zip(future.iterrows(), probs):
            base = float(p * 100)
            horizon = (r["date"].date() - date.today()).days
            # live camera evidence only affects "now" (today / tomorrow)
            score = combine_with_detection(base, density) if horizon <= 1 else base
            rows.append({
                "date": r["date"].date(), "horizon_days": horizon, "location": key,
                "location_name": loc["short"], "audience": loc["audience"],
                "lat": loc["lat"], "lon": loc["lon"],
                "risk_score": round(score, 1), "model_risk": round(base, 1),
                "level": risk_level(score), "drivers": "; ".join(explain_drivers(r)),
                "sst_c": round(r["sst_c"], 1), "wave_height_m": round(r["wave_height_m"], 2),
                "wind_speed_kmh": round(r["wind_speed_kmh"], 1),
                "onshore_wind_kmh": round(r["onshore_wind_kmh"], 1),
                "current_kmh": round(r["current_kmh"], 2), "data_source": r["source"],
            })
    out = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out.to_csv(FORECAST_CSV, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="JellyWatch AI — bloom risk model")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-download data first")
    ap.add_argument("--density", type=float, default=None, help="live camera density at Barakah intake")
    args = ap.parse_args()

    if args.refresh:
        fetch_all()
    if args.train or not RISK_MODEL_PATH.exists():
        train()
    if args.predict or not args.train:
        fc = predict_all(density_by_location={"barakah": args.density} if args.density else None)
        print(f"\n=== Jellyfish bloom risk — today + next {FORECAST_DAYS} days "
              f"(weather: {', '.join(sorted(fc['data_source'].unique()))}; labels: SYNTHETIC) ===")
        for key, g in fc.groupby("location", sort=False):
            loc = LOCATIONS[key]
            print(f"\n{loc['name']}  ({loc['lat']}N, {loc['lon']}E — approx.)")
            for _, r in g.iterrows():
                bar = "#" * int(r["risk_score"] / 5)
                print(f"  {r['date']}  {r['risk_score']:5.1f}  {r['level']:<8} {bar:<20}  {r['drivers']}")
        print(f"\nSaved -> {FORECAST_CSV}")


if __name__ == "__main__":
    main()
