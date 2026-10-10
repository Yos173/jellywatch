"""
Feature engineering + SYNTHETIC bloom labels.

Why synthetic labels?
  Public, day-by-day jellyfish bloom records for the UAE/Gulf coast do not exist.
  So we encode the drivers that marine-biology literature associates with blooms
  as a transparent rule, and use it to generate training labels. The model then
  learns to reproduce that expert rule from REAL weather/ocean inputs.
  As soon as real records (ENEC intake logs, EAD surveys) or camera counts are
  available, they replace `synthetic_bloom_labels` — the pipeline stays the same.

Assumed bloom drivers (hypothesis, NOT ground truth):
  1. Warm water within the favourable range (≈ 24–33 °C) and a warming trend
  2. Calm sea (low waves) for several days -> jellyfish aggregate near surface
  3. Onshore wind pushes surface water (and jellyfish) toward the coast/intake
  4. Weak currents (less dispersion)
  5. Seasonality: spring–summer is the main bloom season in the Gulf
"""
import numpy as np
import pandas as pd

FEATURES = [
    "sst_c", "sst_3d_mean", "sst_7d_change", "sst_anomaly",
    "wave_height_m", "wave_3d_mean", "calm_streak_days",
    "wind_speed_kmh", "onshore_wind_kmh", "onshore_wind_3d_mean",
    "current_kmh", "current_3d_mean",
    "radiation_mj", "month_sin", "month_cos",
]

FEATURE_LABELS = {
    "sst_c": "Sea surface temp", "sst_3d_mean": "SST 3-day mean", "sst_7d_change": "SST 7-day warming",
    "sst_anomaly": "SST anomaly vs month", "wave_height_m": "Wave height", "wave_3d_mean": "Waves 3-day mean",
    "calm_streak_days": "Consecutive calm days", "wind_speed_kmh": "Wind speed",
    "onshore_wind_kmh": "Onshore wind", "onshore_wind_3d_mean": "Onshore wind 3-day",
    "current_kmh": "Ocean current", "current_3d_mean": "Current 3-day mean",
    "radiation_mj": "Solar radiation", "month_sin": "Season (sin)", "month_cos": "Season (cos)",
}

CALM_WAVE_M = 0.5   # a day with max waves below this is "calm"


def build_features(df: pd.DataFrame, onshore_from_deg: float, sst_climatology: dict | None = None) -> pd.DataFrame:
    """Add model features to a daily environment dataframe (sorted by date)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    # Rolling context (what happened over the last few days)
    d["sst_3d_mean"] = d["sst_c"].rolling(3, min_periods=1).mean()
    d["sst_7d_change"] = d["sst_c"] - d["sst_c"].shift(7)
    d["wave_3d_mean"] = d["wave_height_m"].rolling(3, min_periods=1).mean()
    d["current_3d_mean"] = d["current_kmh"].rolling(3, min_periods=1).mean()

    # Onshore component of the wind: + = blowing toward the coast, - = offshore
    angle = np.deg2rad(d["wind_dir_deg"] - onshore_from_deg)
    d["onshore_wind_kmh"] = d["wind_speed_kmh"] * np.cos(angle)
    d["onshore_wind_3d_mean"] = d["onshore_wind_kmh"].rolling(3, min_periods=1).mean()

    # Number of consecutive calm days up to and including today
    calm = (d["wave_height_m"] < CALM_WAVE_M).astype(int)
    d["calm_streak_days"] = calm.groupby((calm == 0).cumsum()).cumsum()

    # Seasonality + anomaly vs the location's monthly climatology
    month = d["date"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * month / 12)
    d["month_cos"] = np.cos(2 * np.pi * month / 12)
    clim = sst_climatology or d.groupby(month)["sst_c"].mean().to_dict()
    d["sst_anomaly"] = d["sst_c"] - month.map(clim)

    d[FEATURES] = d[FEATURES].fillna(0.0)
    return d


def sst_climatology(history: pd.DataFrame) -> dict:
    h = history.copy()
    return h.groupby(pd.to_datetime(h["date"]).dt.month)["sst_c"].mean().to_dict()


def bloom_propensity(d: pd.DataFrame) -> pd.Series:
    """Expert rule -> probability of a bloom (0..1). Transparent and explainable."""
    temp_ok = np.exp(-((d["sst_c"] - 30.0) / 3.5) ** 2)                 # peaks ~30 °C
    warming = np.clip(d["sst_7d_change"] / 2.0, -1, 1)                   # +2 °C in a week = max
    calm = np.clip(d["calm_streak_days"] / 5.0, 0, 1)                    # 5+ calm days = max
    onshore = np.clip(d["onshore_wind_3d_mean"] / 20.0, -1, 1)           # 20 km/h onshore = max
    weak_current = np.clip(1 - d["current_3d_mean"] / 1.5, 0, 1)
    month = d["date"].dt.month
    season = np.where(month.isin([4, 5, 6, 7, 8]), 1.0, np.where(month.isin([3, 9, 10]), 0.5, 0.0))

    z = (2.4 * temp_ok + 0.9 * warming + 1.6 * calm + 1.5 * onshore
         + 0.6 * weak_current + 1.0 * season - 4.3)
    return pd.Series(1 / (1 + np.exp(-z)), index=d.index)


def synthetic_bloom_labels(d: pd.DataFrame, seed: int = 42) -> pd.Series:
    """SYNTHETIC labels: sample bloom / no-bloom from the expert propensity + noise."""
    rng = np.random.default_rng(seed)
    return pd.Series((rng.random(len(d)) < bloom_propensity(d)).astype(int), index=d.index)


def explain_drivers(row: pd.Series) -> list[str]:
    """Human-readable reasons for a forecast — shown in alerts and the dashboard."""
    reasons = []
    if 26 <= row["sst_c"] <= 33:
        reasons.append(f"warm water {row['sst_c']:.1f}°C")
    elif row["sst_c"] > 33:
        reasons.append(f"very warm water {row['sst_c']:.1f}°C (above bloom optimum)")
    if row["sst_7d_change"] > 0.7:
        reasons.append(f"water warming +{row['sst_7d_change']:.1f}°C/week")
    if row["calm_streak_days"] >= 2:
        reasons.append(f"calm sea {int(row['calm_streak_days'])} days (waves {row['wave_height_m']:.2f} m)")
    if row["onshore_wind_3d_mean"] > 5:
        reasons.append(f"onshore wind {row['onshore_wind_3d_mean']:.0f} km/h")
    elif row["onshore_wind_3d_mean"] < -5:
        reasons.append("offshore wind (pushes jellyfish away)")
    if row["current_3d_mean"] < 0.5:
        reasons.append("weak currents")
    if row["wave_height_m"] > 1.2:
        reasons.append(f"rough sea {row['wave_height_m']:.1f} m (disperses blooms)")
    return reasons or ["no strong bloom drivers"]
