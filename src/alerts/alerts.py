"""
Alert rules: risk forecast + live camera density -> who is alerted, and what to do.

Each alert = {level, audience, location, time, valid_for, title, message, action, source}

Audiences
  plant   -> Plant operators   (cooling-water intake: screens, cleaning, load)
  beach   -> Beach managers    (swimmer safety)
  fishing -> Fishermen         (nets, zones)

Usage:
    python src/alerts/alerts.py                 # alerts from latest forecast
    python src/alerts/alerts.py --density 7.5   # + live camera at Barakah intake
    python src/alerts/alerts.py --telegram      # also send (needs TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
"""
import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DENSITY_BLOOM_THRESHOLDS, LOCATIONS  # noqa: E402

LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# (location type, level) -> (title, recommended action)
PLAYBOOK = {
    ("plant", "Medium"): ("Elevated jellyfish risk near intake",
                          "Inspect travelling screens; put cleaning crew on standby; check differential pressure trend."),
    ("plant", "High"): ("High jellyfish risk at intake",
                        "Prepare screens and increase cleaning frequency; deploy surface booms/bubble curtain; "
                        "brief control room on possible load reduction."),
    ("plant", "Critical"): ("CRITICAL: jellyfish bloom expected at intake",
                            "Continuous screen washing; deploy all barriers; pre-plan controlled power reduction; "
                            "notify grid operator."),
    ("beach", "Medium"): ("Jellyfish possible at beach",
                          "Inform lifeguards; keep vinegar/first-aid kits ready; post information signs."),
    ("beach", "High"): ("Swimming caution advised",
                        "Raise purple flag; lifeguards patrol shoreline; publish advisory on city channels."),
    ("beach", "Critical"): ("Beach swimming closure recommended",
                            "Close swimming zone; stinger nets if available; push notification to visitors."),
    ("fishing", "Medium"): ("Jellyfish aggregations possible",
                            "Check nets more often; carry protective gloves."),
    ("fishing", "High"): ("Avoid nearshore zone",
                          "Avoid the nearshore zone around this area; shorten soak times; prefer deeper offshore grounds."),
    ("fishing", "Critical"): ("Avoid zone — dense bloom expected",
                              "Do not set nets in this zone; jellyfish catch can tear nets and spoil the haul. "
                              "Report sightings to help the network."),
}


def _when(horizon: int) -> str:
    return {0: "today", 1: "in ~24h", 2: "in ~48h"}.get(horizon, f"in ~{horizon * 24}h")


def generate_alerts(forecast: pd.DataFrame, live_density: dict | None = None,
                    min_level: str = "Medium") -> list[dict]:
    """Turn forecast rows (+ optional live camera densities per location) into alerts."""
    alerts = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 1) Forecast-based alerts: worst upcoming day per location only (avoid spam)
    for key, g in forecast.groupby("location", sort=False):
        g = g.assign(rank=g["level"].map(LEVEL_RANK)).sort_values(["rank", "risk_score"], ascending=False)
        top = g.iloc[0]
        if LEVEL_RANK[top["level"]] < LEVEL_RANK[min_level]:
            continue
        loc = LOCATIONS[key]
        title, action = PLAYBOOK[(loc["type"], top["level"])]
        horizon = int(top["horizon_days"])
        alerts.append({
            "level": top["level"], "audience": loc["audience"], "location": loc["short"],
            "time": now, "valid_for": str(top["date"]),
            "title": f"{title} {_when(horizon)}",
            "message": f"{loc['name']}: bloom risk {top['risk_score']:.0f}/100 {_when(horizon)}. "
                       f"Drivers: {top['drivers']}.",
            "action": action, "source": "forecast", "risk_score": float(top["risk_score"]),
        })

    # 2) Live camera alerts: what the detector sees right now
    for key, density in (live_density or {}).items():
        if density is None:
            continue
        level = "Low"
        for name, thr in DENSITY_BLOOM_THRESHOLDS.items():
            if density >= thr:
                level = name
        if LEVEL_RANK[level] < LEVEL_RANK[min_level]:
            continue
        loc = LOCATIONS[key]
        title, action = PLAYBOOK[(loc["type"], level)]
        alerts.append({
            "level": level, "audience": loc["audience"], "location": loc["short"],
            "time": now, "valid_for": str(date.today()),
            "title": f"LIVE CAMERA: {title.lower()}",
            "message": f"Camera at {loc['name']} sees {density:.1f} jellyfish per frame (smoothed).",
            "action": action, "source": "camera", "risk_score": None,
        })

    return sorted(alerts, key=lambda a: (-LEVEL_RANK[a["level"]], a["source"] != "camera"))


def format_alert(a: dict) -> str:
    icon = {"Medium": "🟡", "High": "🟠", "Critical": "🔴"}.get(a["level"], "🟢")
    return (f"{icon} [{a['level'].upper()}] {a['title']}\n"
            f"   To: {a['audience']} | Where: {a['location']} | Valid: {a['valid_for']} | Issued: {a['time']}\n"
            f"   {a['message']}\n"
            f"   ➜ Action: {a['action']}")


def send_telegram(alerts: list[dict]) -> int:
    """Optional: push alerts to a Telegram chat. Needs env vars TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID."""
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID).")
        return 0
    sent = 0
    for a in alerts:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": "JellyWatch AI\n" + format_alert(a)}, timeout=15)
        sent += r.ok
    return sent


def main() -> None:
    from src.prediction.risk_model import FORECAST_CSV, predict_all

    ap = argparse.ArgumentParser(description="JellyWatch AI — alerts")
    ap.add_argument("--density", type=float, default=None, help="live camera density at Barakah intake")
    ap.add_argument("--min-level", default="Medium", choices=list(LEVEL_RANK))
    ap.add_argument("--telegram", action="store_true")
    args = ap.parse_args()

    live = {"barakah": args.density} if args.density else None
    fc = predict_all(density_by_location=live) if not FORECAST_CSV.exists() or live else pd.read_csv(FORECAST_CSV)
    alerts = generate_alerts(fc, live, args.min_level)
    print(f"=== {len(alerts)} JellyWatch alerts ===\n")
    for a in alerts:
        print(format_alert(a) + "\n")
    if args.telegram:
        print(f"Sent {send_telegram(alerts)} Telegram messages.")


if __name__ == "__main__":
    main()
