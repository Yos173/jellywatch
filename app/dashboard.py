"""
JellyWatch AI — Streamlit dashboard (what the judges see).

Panels
  1. Map of the UAE coast — monitored sites coloured by bloom risk
  2. Live detection — run the YOLO detector on demo / uploaded video
  3. Forecast — 3-day bloom risk per site + model explainability
  4. Alerts feed — who must act, and how

Run:  streamlit run app/dashboard.py
"""
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from src.alerts.alerts import generate_alerts  # noqa: E402
from src.config import (DENSITY_BLOOM_THRESHOLDS, DETECT_CONF, DETECTOR_WEIGHTS, ENV_DIR,  # noqa: E402
                        LOCATIONS, MODELS_DIR, OUTPUTS_DIR, RISK_LEVELS, VIDEOS_DIR, level_color)

st.set_page_config(page_title="JellyWatch AI", page_icon=str(ROOT / "assets" / "logo.png"), layout="wide")

TEAM = "Blue Sentinel"
# map label offsets so nearby sites (Barakah / Ruwais) don't overlap
LABEL_SHIFT = {"barakah": "translate(-125px,12px)", "ruwais": "translate(14px,-22px)"}
PLOT_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(color="#cfe6ff"), margin=dict(l=10, r=10, t=30, b=10))

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
.stApp {background: radial-gradient(1200px 700px at 8% -10%, #0e3a5c 0%, #04121f 60%) fixed;}
.block-container {padding-top: 3.6rem; max-width: 1500px;}
.jw-header {display:flex; align-items:center; gap:18px; padding:16px 22px; border-radius:18px;
  background: linear-gradient(90deg, rgba(14,58,92,.92), rgba(4,21,38,.55));
  border:1px solid rgba(34,211,238,.25); box-shadow: 0 8px 30px rgba(0,0,0,.35);}
.jw-header h1 {margin:0; padding:0; font-size:2.1rem; letter-spacing:.5px; color:#e6fbff;}
.jw-header p {margin:2px 0 0 0; color:#9cc9e6; font-size:.98rem;}
.jw-tag {margin-left:auto; text-align:right; color:#9cc9e6; font-size:.85rem;}
.kpi {padding:14px 16px; border-radius:14px; background:rgba(11,34,54,.85);
  border:1px solid rgba(34,211,238,.18); height:100%;}
.kpi .lbl {color:#8fb7d4; font-size:.78rem; text-transform:uppercase; letter-spacing:.8px;}
.kpi .val {font-size:1.7rem; font-weight:700; color:#e6fbff; line-height:1.25;}
.kpi .sub {color:#8fb7d4; font-size:.8rem;}
.alert-card {border-left:6px solid #888; background:rgba(11,34,54,.9); border-radius:12px;
  padding:10px 14px; margin-bottom:10px;}
.alert-card .t {font-weight:700; color:#fff;}
.alert-card .m {color:#cfe6ff; font-size:.88rem; margin:4px 0;}
.alert-card .a {color:#a5f3fc; font-size:.88rem;}
.alert-card .meta {color:#8fb7d4; font-size:.75rem;}
.badge {display:inline-block; padding:1px 8px; border-radius:999px; font-size:.72rem; font-weight:700;}
.b-real {background:#064e3b; color:#a7f3d0;} .b-syn {background:#7c2d12; color:#fed7aa;}
.b-demo {background:#1e3a8a; color:#bfdbfe;}
h3 {color:#e6fbff !important;}
</style>
""", unsafe_allow_html=True)


def logo_b64() -> str:
    return base64.b64encode((ROOT / "assets" / "logo.svg").read_bytes()).decode()


def badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{text}</span>'


def kpi(col, label: str, value: str, sub: str = "", color: str | None = None) -> None:
    style = f' style="color:{color}"' if color else ""
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val"{style}>{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Computing bloom risk forecast ...")
def get_forecast(density_items: tuple, refresh_token: int) -> pd.DataFrame:
    from src.prediction.risk_model import predict_all
    return predict_all(density_by_location=dict(density_items) or None)


@st.cache_data(show_spinner=False)
def get_backtest() -> pd.DataFrame:
    """Model risk over the last 365 days of REAL weather (shows the seasonal pattern)."""
    from src.prediction.features import FEATURES, build_features
    from src.prediction.risk_model import load_model
    model = load_model()["model"]
    frames = []
    for key, loc in LOCATIONS.items():
        hist = pd.read_csv(ENV_DIR / f"{key}_history.csv")
        d = build_features(hist, loc["onshore_from_deg"]).tail(365)
        d["risk"] = model.predict_proba(d[FEATURES])[:, 1] * 100
        d["risk_7d"] = d["risk"].rolling(7, min_periods=1).mean()
        d["location"] = loc["short"]
        frames.append(d[["date", "location", "risk_7d", "sst_c"]])
    return pd.concat(frames)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


if "live_density" not in st.session_state:
    st.session_state.live_density = {}
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if "det_series" not in st.session_state:
    st.session_state.det_series = None
    st.session_state.det_frame = None

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.image(str(ROOT / "assets" / "logo.png"), width=120)
    st.markdown(f"### JellyWatch AI\nTeam **{TEAM}** · Yosef")
    st.divider()
    st.markdown("**Live camera**")
    cam_loc = st.selectbox("Camera installed at", list(LOCATIONS),
                           format_func=lambda k: LOCATIONS[k]["short"], index=0)
    conf = st.slider("Detection confidence", 0.10, 0.80, DETECT_CONF, 0.05)
    manual = st.slider("Manual camera density override (0 = off)", 0.0, 15.0, 0.0, 0.5,
                       help="Simulate what the intake camera sees, without running a video.")
    if manual > 0:
        st.session_state.live_density[cam_loc] = manual
    if st.button("Clear camera evidence", width="stretch"):
        st.session_state.live_density = {}
        st.session_state.det_series = None
        st.rerun()
    st.divider()
    if st.button("🔄 Refresh ocean & weather data", width="stretch"):
        from src.prediction.fetch_data import fetch_all
        with st.spinner("Downloading Open-Meteo marine + weather forecast ..."):
            fetch_all(history=False, forecast=True)
        st.session_state.refresh += 1
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("**Data provenance**")
    st.markdown(f"{badge('REAL', 'real')} Ocean & weather — Open-Meteo<br>"
                f"{badge('REAL', 'real')} Jellyfish images — Aquarium, Brackish, Open Images<br>"
                f"{badge('SYNTHETIC', 'syn')} Bloom labels — expert rule<br>"
                f"{badge('DEMO', 'demo')} Video — Wikimedia Commons footage<br>"
                f"{badge('APPROX', 'demo')} Site coordinates", unsafe_allow_html=True)

density_items = tuple(sorted(st.session_state.live_density.items()))
forecast = get_forecast(density_items, st.session_state.refresh)
alerts = generate_alerts(forecast, st.session_state.live_density)
det_metrics = read_json(MODELS_DIR / "detector_metrics.json")
risk_metrics = read_json(MODELS_DIR / "risk_metrics.json")

# --------------------------------------------------------------------------- header
st.markdown(f"""
<div class="jw-header">
  <img src="data:image/svg+xml;base64,{logo_b64()}" width="74">
  <div><h1>JellyWatch AI</h1>
  <p>Detect · Predict · Alert — early warning for jellyfish blooms on the UAE coast</p></div>
  <div class="jw-tag">ATRC ChallengeOn · ENEC challenge<br>Team <b>{TEAM}</b></div>
</div>""", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------- KPIs
today = forecast[forecast["horizon_days"] == 0]
worst = forecast.loc[forecast["risk_score"].idxmax()]
barakah_now = today[today["location"] == "barakah"].iloc[0]
cam_density = st.session_state.live_density.get(cam_loc)
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Barakah intake — today", f"{barakah_now['risk_score']:.0f}/100",
    barakah_now["level"], level_color(barakah_now["level"]))
kpi(k2, "Highest risk (3 days)", f"{worst['risk_score']:.0f}/100",
    f"{worst['location_name']} · {worst['date']}", level_color(worst["level"]))
kpi(k3, "Live camera density", f"{cam_density:.1f}" if cam_density else "—",
    f"jellyfish/frame · {LOCATIONS[cam_loc]['short']}")
kpi(k4, "Active alerts", str(len(alerts)),
    f"{sum(a['level'] in ('High', 'Critical') for a in alerts)} high / critical")
kpi(k5, "Detector mAP@50 (test)", f"{det_metrics.get('test', {}).get('mAP50', 0):.2f}" if det_metrics else "—",
    f"risk model ROC-AUC {risk_metrics.get('roc_auc', '—')}")
st.write("")

# --------------------------------------------------------------------------- 1. map + 4. alerts
c_map, c_alert = st.columns([3, 2], gap="medium")
with c_map:
    st.markdown("### 🗺️ Coastal risk map")
    # Esri basemaps need no API key
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    m = folium.Map(location=[24.25, 53.3], zoom_start=8, tiles=None, control_scale=True)
    # the layer added LAST is shown by default -> satellite
    folium.TileLayer(esri.format("Ocean/World_Ocean_Base"), attr="Esri, GEBCO, NOAA", name="Ocean").add_to(m)
    folium.TileLayer(esri.format("World_Imagery"), attr="Esri, Maxar", name="Satellite").add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    lats = [loc["lat"] for loc in LOCATIONS.values()]
    lons = [loc["lon"] for loc in LOCATIONS.values()]
    m.fit_bounds([[min(lats) - 0.35, min(lons) - 0.5], [max(lats) + 0.35, max(lons) + 0.9]])
    for key, loc in LOCATIONS.items():
        g = forecast[forecast["location"] == key]
        now = g[g["horizon_days"] == 0].iloc[0]
        peak = g.loc[g["risk_score"].idxmax()]
        color = level_color(now["level"])
        rows = "".join(f"<tr><td>{r['date']}</td><td><b style='color:{level_color(r['level'])}'>"
                       f"{r['risk_score']:.0f}</b></td><td>{r['level']}</td></tr>" for _, r in g.iterrows())
        popup = (f"<div style='font-family:sans-serif;width:260px'><b>{loc['name']}</b><br>"
                 f"<small>~{loc['lat']}N, {loc['lon']}E (approximate)</small><br>"
                 f"Audience: {loc['audience']}<table style='width:100%;margin-top:6px'>{rows}</table>"
                 f"<small>Drivers today: {now['drivers']}</small></div>")
        folium.Circle([loc["lat"], loc["lon"]], radius=6000 + 180 * peak["risk_score"], color=color,
                      fill=True, fill_opacity=0.18, weight=1).add_to(m)
        folium.CircleMarker([loc["lat"], loc["lon"]], radius=11, color="#ffffff", weight=2, fill=True,
                            fill_color=color, fill_opacity=0.95, popup=folium.Popup(popup, max_width=300),
                            tooltip=f"{loc['short']}: {now['risk_score']:.0f}/100 ({now['level']})").add_to(m)
        folium.Marker([loc["lat"], loc["lon"]], icon=folium.DivIcon(
            html=f"<div style='color:#fff;font-weight:700;font-size:12px;white-space:nowrap;"
                 f"text-shadow:0 0 4px #000;transform:{LABEL_SHIFT.get(key, 'translate(16px,-8px)')}'>{loc['short']} · "
                 f"{now['risk_score']:.0f}</div>")).add_to(m)
    st_folium(m, height=430, use_container_width=True, returned_objects=[])
    legend = " ".join(f"<span style='color:{c}'>●</span> {n} ({lo}–{min(hi, 100)})" for lo, hi, n, c in RISK_LEVELS)
    st.caption(f"{legend} · circle size = 3-day peak risk · coordinates are approximate", unsafe_allow_html=True)

with c_alert:
    st.markdown("### 🚨 Alerts feed")
    if not alerts:
        st.success("No alerts at Medium level or above. Routine monitoring.")
    for a in alerts[:6]:
        color = level_color(a["level"])
        src = "📷 camera" if a["source"] == "camera" else "📈 forecast"
        st.markdown(f"""<div class="alert-card" style="border-left-color:{color}">
          <div class="t"><span style="color:{color}">{a['level'].upper()}</span> · {a['title']}</div>
          <div class="meta">To <b>{a['audience']}</b> · {a['location']} · valid {a['valid_for']} · {src} · issued {a['time']}</div>
          <div class="m">{a['message']}</div>
          <div class="a">➜ {a['action']}</div></div>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- 2. live detection
st.markdown("### 🎥 Live detection — camera at " + LOCATIONS[cam_loc]["name"])
demo_videos = sorted(VIDEOS_DIR.glob("demo_*.mp4"), key=lambda p: p.name != "demo_bloom.mp4")
c_ctrl, c_vid, c_stats = st.columns([1.1, 2.6, 1.6], gap="medium")
with c_ctrl:
    choice = st.radio("Source", [v.name for v in demo_videos] + ["Upload video / image"], index=0)
    path = None
    if choice == "Upload video / image":
        up = st.file_uploader("Video or image", type=["mp4", "mov", "avi", "webm", "mkv", "jpg", "jpeg", "png"])
        if up:
            up_dir = OUTPUTS_DIR / "uploads"
            up_dir.mkdir(parents=True, exist_ok=True)
            path = up_dir / Path(up.name).name
            path.write_bytes(up.getbuffer())
    else:
        path = VIDEOS_DIR / choice
    stride = st.select_slider("Process every Nth frame", [1, 2, 3, 4], value=2)
    max_frames = st.slider("Max frames", 50, 600, 250, 50)
    run = st.button("▶ Run detection", type="primary", width="stretch",
                    disabled=path is None or not DETECTOR_WEIGHTS.exists())
    if not DETECTOR_WEIGHTS.exists():
        st.warning("Train the detector first: `python src/detection/train.py`")
    st.caption("Demo footage: moon-jellyfish bloom (Wikimedia Commons, CC BY-SA). "
               "Not filmed at Barakah — stands in for an intake camera.")

frame_ph = c_vid.empty()
with c_stats:
    m1, m2 = st.columns(2)
    count_ph, dens_ph = m1.empty(), m2.empty()
    m3, m4 = st.columns(2)
    uniq_ph, lvl_ph = m3.empty(), m4.empty()
    chart_ph = st.empty()


def draw_stats(series: pd.DataFrame) -> None:
    last = series.iloc[-1]
    count_ph.metric("In frame", int(last["count"]))
    dens_ph.metric("Density", f"{last['density']:.1f}")
    uniq_ph.metric("Unique tracked", int(last["unique_tracked"]))
    lvl_ph.markdown(f"<div class='kpi'><div class='lbl'>Camera level</div><div class='val' "
                    f"style='color:{level_color(last['level'])};font-size:1.3rem'>{last['level']}</div></div>",
                    unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(x=series["video_time_s"], y=series["count"], name="count", marker_color="rgba(34,211,238,.35)")
    fig.add_scatter(x=series["video_time_s"], y=series["density"], name="density (smoothed)",
                    line=dict(color="#facc15", width=3))
    for name, thr in DENSITY_BLOOM_THRESHOLDS.items():
        fig.add_hline(y=thr, line_dash="dot", line_color=level_color(name), opacity=.6,
                      annotation_text=name, annotation_font_color=level_color(name))
    fig.update_layout(height=260, **PLOT_BG, legend=dict(orientation="h", y=1.15), xaxis_title="video time (s)",
                      yaxis_title="jellyfish / frame")
    chart_ph.plotly_chart(fig, width="stretch", key=f"det_{time.time_ns()}")


if run and path is not None:
    from src.detection.detect import JellyDetector, stream
    detector = JellyDetector(conf=conf)
    rows = []
    t0 = time.time()
    for i, (frame, row) in enumerate(stream(str(path), detector, stride=stride, max_frames=max_frames)):
        rows.append(row)
        frame_ph.image(frame[:, :, ::-1], width="stretch",
                       caption=f"frame {row['frame']} · {len(rows) / max(time.time() - t0, 1e-6):.1f} fps")
        if i % 5 == 0:
            draw_stats(pd.DataFrame(rows))
        st.session_state.det_frame = frame[:, :, ::-1]
    series = pd.DataFrame(rows)
    st.session_state.det_series = series
    if len(series):
        # Robust "what the camera saw": 90th percentile of the smoothed density
        st.session_state.live_density[cam_loc] = round(float(series["density"].quantile(0.9)), 2)
        series.to_csv(OUTPUTS_DIR / "dashboard_detections.csv", index=False)
    st.rerun()   # recompute forecast + alerts with the new camera evidence

if st.session_state.det_series is not None and len(st.session_state.det_series):
    frame_ph.image(st.session_state.det_frame, width="stretch", caption="last processed frame")
    draw_stats(st.session_state.det_series)
    c_vid.success(f"Camera evidence fed into the forecast: density "
                  f"{st.session_state.live_density.get(cam_loc, 0):.1f} jellyfish/frame at "
                  f"{LOCATIONS[cam_loc]['short']} → risk and alerts updated above.")
else:
    frame_ph.info("Press **Run detection** to analyse the selected footage.")

# --------------------------------------------------------------------------- 3. forecast
st.markdown("### 📈 Bloom risk forecast — today + next 3 days")
c_fc, c_tab = st.columns([3, 2], gap="medium")
with c_fc:
    fig = go.Figure()
    for lo, hi, name, color in RISK_LEVELS:
        fig.add_hrect(y0=lo, y1=min(hi, 100), fillcolor=color, opacity=0.07, line_width=0,
                      annotation_text=name, annotation_position="right", annotation_font_color=color)
    palette = {"barakah": "#22d3ee", "corniche": "#a78bfa", "ruwais": "#f472b6"}
    for key, g in forecast.groupby("location", sort=False):
        fig.add_scatter(x=g["date"].astype(str), y=g["risk_score"], mode="lines+markers+text",
                        name=LOCATIONS[key]["short"], line=dict(color=palette[key], width=3),
                        marker=dict(size=10), text=g["risk_score"].round(0).astype(int),
                        textposition="top center", textfont=dict(color=palette[key]))
    fig.update_layout(height=360, yaxis=dict(range=[0, 105], title="risk score (0–100)"),
                      xaxis=dict(type="category"), **PLOT_BG,
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
with c_tab:
    tab = forecast[["date", "location_name", "risk_score", "level", "sst_c", "wave_height_m",
                    "onshore_wind_kmh", "drivers"]].rename(columns={
        "location_name": "site", "risk_score": "risk", "sst_c": "SST °C", "wave_height_m": "waves m",
        "onshore_wind_kmh": "onshore km/h"})
    st.dataframe(tab, height=360, hide_index=True, width="stretch",
                 column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100,
                                                                         format="%.0f")})

c_imp, c_hist = st.columns([2, 3], gap="medium")
with c_imp:
    st.markdown("#### Why? — what drives the model")
    imp_path = MODELS_DIR / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["label"], orientation="h",
                               marker_color="#22d3ee"))
        fig.update_layout(height=330, **PLOT_BG, xaxis_title="feature importance")
        st.plotly_chart(fig, width="stretch")
with c_hist:
    st.markdown("#### Last 12 months — model risk on real weather (7-day mean)")
    bt = get_backtest()
    fig = go.Figure()
    for loc_name, g in bt.groupby("location", sort=False):
        fig.add_scatter(x=g["date"], y=g["risk_7d"], name=loc_name, mode="lines")
    fig.update_layout(height=330, **PLOT_BG, yaxis=dict(range=[0, 100], title="risk"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- method / honesty
with st.expander("ℹ️ How it works · what is real vs synthetic"):
    st.markdown(f"""
**Pipeline:** camera frames → **YOLO11s detector** (fine-tuned on jellyfish) → count + smoothed density →
blended with a **Gradient-Boosting risk model** fed by REAL Open-Meteo marine & weather data
(sea-surface temperature, waves, currents, wind direction relative to each coast, season) → rule-based **alert playbook**
per audience (plant operators, beach managers, fishermen).

| Component | Status |
|---|---|
| Ocean / weather inputs (history since 2022 + 4-day forecast) | **REAL** — Open-Meteo Marine & Weather APIs |
| Detector training images | **REAL** — Roboflow *Aquarium*, Aalborg Univ. *Brackish* fixed underwater camera, Google *Open Images V7* (CC BY) |
| Detector accuracy | test mAP@50 = {det_metrics.get('test', {}).get('mAP50', 'n/a')}, val mAP@50 = {det_metrics.get('val', {}).get('mAP50', 'n/a')} |
| Bloom labels for the risk model | **SYNTHETIC** — expert rule (warm water, calm seas, onshore wind, weak currents, season) + noise. No public daily Gulf bloom records exist. |
| Risk model quality | ROC-AUC {risk_metrics.get('roc_auc', 'n/a')} vs the synthetic labels on the most recent 12 months — measures fidelity to the rule, *not* real-world skill |
| Demo video | Wikimedia Commons moon-jellyfish footage (Netherlands) — stand-in for an intake camera |
| Site coordinates | Approximate |

**Next step to make it operational:** replace synthetic labels with ENEC intake-screen logs / EAD survey data and a real
camera at the Barakah intake — the code path stays identical.
""")
st.caption(f"JellyWatch AI · Team {TEAM} · ATRC ChallengeOn — ENEC challenge · prototype, not for operational use")
