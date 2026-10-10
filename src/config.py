"""
JellyWatch AI — central configuration.

Everything that a judge might ask "where does this number come from?" lives here:
monitored locations, risk thresholds, file paths and model settings.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"            # YOLO datasets
VIDEOS_DIR = DATA_DIR / "videos"            # demo footage
ENV_DIR = DATA_DIR / "environment"          # ocean + weather CSVs
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"              # annotated videos, detection CSVs

DATASET_YAML = IMAGES_DIR / "jellyfish" / "data.yaml"
DETECTOR_WEIGHTS = MODELS_DIR / "best.pt"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
DEMO_VIDEO = VIDEOS_DIR / "demo_bloom.mp4"

# ---------------------------------------------------------------------------
# Monitored locations (APPROXIMATE coordinates — shown as such in the UI)
# onshore_from_deg = compass direction the wind blows FROM when it pushes
# water (and jellyfish) toward this coast. UAE Gulf coast faces roughly N/NW.
# ---------------------------------------------------------------------------
LOCATIONS = {
    "barakah": {
        "name": "Barakah NPP — cooling water intake",
        "short": "Barakah intake",
        "lat": 23.96, "lon": 52.26,
        "type": "plant",
        "audience": "Plant operators",
        "onshore_from_deg": 340,
    },
    "corniche": {
        "name": "Abu Dhabi Corniche public beach",
        "short": "Corniche beach",
        "lat": 24.48, "lon": 54.35,
        "type": "beach",
        "audience": "Beach managers",
        "onshore_from_deg": 320,
    },
    "ruwais": {
        "name": "Ruwais coastal fishing grounds",
        "short": "Ruwais fishing",
        "lat": 24.11, "lon": 52.73,
        "type": "fishing",
        "audience": "Fishermen",
        "onshore_from_deg": 350,
    },
}
COORDS_ARE_APPROXIMATE = True

# ---------------------------------------------------------------------------
# Risk levels (score 0–100 -> level). Upper bounds are exclusive.
# ---------------------------------------------------------------------------
RISK_LEVELS = [
    (0, 30, "Low", "#2ecc71"),
    (30, 55, "Medium", "#f1c40f"),
    (55, 80, "High", "#e67e22"),
    (80, 101, "Critical", "#e74c3c"),
]


def risk_level(score: float) -> str:
    """Map a 0–100 risk score to Low / Medium / High / Critical."""
    for lo, hi, name, _ in RISK_LEVELS:
        if lo <= score < hi:
            return name
    return "Critical"


def level_color(level: str) -> str:
    return {name: color for _, _, name, color in RISK_LEVELS}.get(level, "#95a5a6")


# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
BASE_YOLO_MODEL = "yolo11s.pt"     # small model; fine-tuned on jellyfish
DETECT_CONF = 0.30                 # minimum confidence for a detection
DENSITY_WINDOW = 15                # frames used for the smoothed density score

# Live camera density (jellyfish per frame) that we treat as a bloom signal.
# Used to boost the forecast risk when a camera actually sees jellyfish.
DENSITY_BLOOM_THRESHOLDS = {"Medium": 2.0, "High": 5.0, "Critical": 10.0}

FORECAST_DAYS = 3
