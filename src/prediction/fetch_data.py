"""
Pull REAL daily ocean + weather data for each monitored UAE location.

Sources (free, no API key):
  * Open-Meteo Marine API   -> sea surface temperature, wave height, ocean current
  * Open-Meteo Archive API  -> historical wind speed/direction, air temp, sunshine
  * Open-Meteo Forecast API -> next days' wind etc. (+ last 14 days for trends)

If the internet is unavailable, a clearly-labelled SYNTHETIC series is generated
instead (column `source` = "synthetic") so the demo never breaks.

Output CSVs in data/environment/:
  <loc>_history.csv   ~3 years of daily data (used to train the risk model)
  <loc>_forecast.csv  last 14 days + next 4 days (used to predict)
Columns: date, sst_c, wave_height_m, current_kmh, current_dir_deg,
         wind_speed_kmh, wind_dir_deg, air_temp_c, radiation_mj, source, is_forecast

Usage:
    python src/prediction/fetch_data.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import ENV_DIR, LOCATIONS  # noqa: E402

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TZ = "Asia/Dubai"
HISTORY_START = date(2022, 1, 1)
WEATHER_DAILY = "wind_speed_10m_max,wind_direction_10m_dominant,temperature_2m_max,shortwave_radiation_sum"
MARINE_HOURLY = "sea_surface_temperature,ocean_current_velocity,ocean_current_direction"
COLUMNS = ["date", "sst_c", "wave_height_m", "current_kmh", "current_dir_deg", "wind_speed_kmh",
           "wind_dir_deg", "air_temp_c", "radiation_mj", "source", "is_forecast"]


def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params={**params, "timezone": TZ}, timeout=60)
    r.raise_for_status()
    return r.json()


def _circular_mean(deg: pd.Series) -> float:
    rad = np.deg2rad(deg.dropna())
    if rad.empty:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def _marine_daily(lat: float, lon: float, **range_params) -> pd.DataFrame:
    js = _get(MARINE_URL, {"latitude": lat, "longitude": lon, "hourly": MARINE_HOURLY,
                           "daily": "wave_height_max", **range_params})
    h = pd.DataFrame(js["hourly"])
    h["date"] = pd.to_datetime(h["time"]).dt.date
    daily = h.groupby("date").agg(
        sst_c=("sea_surface_temperature", "mean"),
        current_kmh=("ocean_current_velocity", "mean"),
        current_dir_deg=("ocean_current_direction", _circular_mean),
    )
    waves = pd.DataFrame(js["daily"]).rename(columns={"time": "date", "wave_height_max": "wave_height_m"})
    waves["date"] = pd.to_datetime(waves["date"]).dt.date
    return daily.reset_index().merge(waves, on="date", how="outer")


def _weather_daily(url: str, lat: float, lon: float, **range_params) -> pd.DataFrame:
    js = _get(url, {"latitude": lat, "longitude": lon, "daily": WEATHER_DAILY, **range_params})
    d = pd.DataFrame(js["daily"]).rename(columns={
        "time": "date", "wind_speed_10m_max": "wind_speed_kmh", "wind_direction_10m_dominant": "wind_dir_deg",
        "temperature_2m_max": "air_temp_c", "shortwave_radiation_sum": "radiation_mj"})
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d


def fetch_history(loc: dict) -> pd.DataFrame:
    end = date.today() - timedelta(days=7)          # archive has a few days' delay
    rng = {"start_date": HISTORY_START.isoformat(), "end_date": end.isoformat()}
    df = _marine_daily(loc["lat"], loc["lon"], **rng).merge(
        _weather_daily(ARCHIVE_URL, loc["lat"], loc["lon"], **rng), on="date", how="inner")
    df["source"], df["is_forecast"] = "open-meteo", False
    return df


def fetch_forecast(loc: dict, past_days: int = 14, forecast_days: int = 4) -> pd.DataFrame:
    rng = {"past_days": past_days, "forecast_days": forecast_days}
    df = _marine_daily(loc["lat"], loc["lon"], **rng).merge(
        _weather_daily(FORECAST_URL, loc["lat"], loc["lon"], **rng), on="date", how="inner")
    df["source"] = "open-meteo"
    df["is_forecast"] = pd.to_datetime(df["date"]) >= pd.Timestamp(date.today())
    return df


# --------------------------------------------------------------------- fallback
def synthetic_series(start: date, end: date, seed: int, today: date | None = None) -> pd.DataFrame:
    """Plausible Gulf-like weather, used ONLY when the API is unreachable. Labelled 'synthetic'."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 120) / 365)          # peak ~ early August
    n = len(dates)
    df = pd.DataFrame({
        "date": dates.date,
        "sst_c": 27 + 6.5 * season + rng.normal(0, 0.6, n).cumsum() * 0.05 + rng.normal(0, 0.3, n),
        "wave_height_m": np.clip(rng.gamma(2.0, 0.22, n), 0.05, 3),
        "current_kmh": np.clip(rng.gamma(2.0, 0.25, n), 0.0, 3),
        "current_dir_deg": rng.uniform(0, 360, n),
        "wind_speed_kmh": np.clip(rng.gamma(4.0, 5.0, n), 2, 70),
        "wind_dir_deg": (330 + rng.normal(0, 60, n)) % 360,
        "air_temp_c": 33 + 9 * season + rng.normal(0, 1.5, n),
        "radiation_mj": 20 + 7 * season + rng.normal(0, 1.5, n),
    })
    df["source"] = "synthetic"
    df["is_forecast"] = pd.to_datetime(df["date"]) >= pd.Timestamp(today or date.today())
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    df = df[COLUMNS].sort_values("date")
    num = df.select_dtypes("number").columns
    df[num] = df[num].interpolate(limit_direction="both").round(3)   # fill small gaps
    df.to_csv(path, index=False)


def fetch_all(history: bool = True, forecast: bool = True) -> dict:
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    status = {}
    for i, (key, loc) in enumerate(LOCATIONS.items()):
        for kind, enabled in (("history", history), ("forecast", forecast)):
            if not enabled:
                continue
            path = ENV_DIR / f"{key}_{kind}.csv"
            try:
                df = fetch_history(loc) if kind == "history" else fetch_forecast(loc)
                src = "open-meteo (REAL)"
            except Exception as e:  # network / API problem -> synthetic fallback
                print(f"  ! {key} {kind}: API failed ({e}); using SYNTHETIC data")
                if kind == "history":
                    df = synthetic_series(HISTORY_START, date.today() - timedelta(days=7), seed=i)
                else:
                    df = synthetic_series(date.today() - timedelta(days=14), date.today() + timedelta(days=3), seed=100 + i)
                src = "SYNTHETIC fallback"
            save(df, path)
            status[f"{key}_{kind}"] = src
            print(f"  {key:<9} {kind:<8} {len(df):>5} days  [{src}] -> {path.name}")
    return status


if __name__ == "__main__":
    print("Fetching ocean + weather data for UAE locations ...")
    fetch_all()
