"""Configuration settings for the orbital collision predictor."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, SAVED_MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# TLE Data Sources
TLE_SOURCES = {
    "active_satellites": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "space_stations": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "oneweb": "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
    "iridium": "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle",
    "debris": "https://celestrak.org/NORAD/elements/gp.php?GROUP=1982-092&FORMAT=tle",
    "recent_launches": "https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle"
}

# Model parameters
MODEL_CONFIG = {
    "lstm": {
        "units": 128,
        "dropout": 0.2,
        "sequence_length": 24,  # hours of historical data
        "prediction_horizon": 72,  # hours to predict ahead
        "batch_size": 32,
        "epochs": 50
    },
    "collision_detection": {
        "min_distance_km": 10.0,  # Minimum distance to consider for collision
        "probability_threshold": 0.01,  # 1% probability threshold
        "time_step_minutes": 1,  # Temporal resolution for collision checking
        "monte_carlo_samples": 1000  # Number of samples for uncertainty estimation
    }
}

# Visualization settings
VIZ_CONFIG = {
    "earth_radius_km": 6371.0,
    "color_scheme": {
        "active_satellite": "#00ff00",
        "debris": "#ff0000",
        "space_station": "#0000ff",
        "starlink": "#ff9900",
        "collision_risk": "#ff00ff"
    },
    "update_interval_seconds": 60
}

# API Rate limiting
API_CONFIG = {
    "requests_per_minute": 10,
    "cache_expiry_hours": 6,
    "retry_attempts": 3,
    "timeout_seconds": 30
}