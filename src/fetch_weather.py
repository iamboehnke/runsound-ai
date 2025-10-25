"""
fetch_weather.py
Fetch weather data for each run stored in data/latest_runs.json
and save to data/run_weather.json.

Uses Open-Meteo API (no key needed).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUNS_PATH = DATA_DIR / "latest_runs.json"
OUT_PATH = DATA_DIR / "run_weather.json"

def fetch_weather(lat: float, lon: float, timestamp: str) -> Dict[str, Any]:
    """Fetch weather near run start using Open-Meteo historical or forecast data."""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y-%m-%d")

    now_utc = datetime.now(timezone.utc)
    if (now_utc - dt).days > 7:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,weathercode,relative_humidity_2m,apparent_temperature,windspeed_10m",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Weather API failed: {resp.status_code} {resp.text}")

    data = resp.json()
    if "hourly" not in data:
        return {}

    hours = data["hourly"]["time"]
    idx = min(
        range(len(hours)),
        key=lambda i: abs(
            datetime.fromisoformat(hours[i]).replace(tzinfo=timezone.utc) - dt
        ),
    )

    return {k: v[idx] for k, v in data["hourly"].items() if k != "time"}


def fetch_weather_for_all_runs():
    if not RUNS_PATH.exists():
        raise FileNotFoundError("latest_runs.json not found. Run fetch_strava.py first.")

    runs = json.loads(RUNS_PATH.read_text())
    weather_data = []

    for run in runs:
        start_latlng = run.get("start_latlng")
        start_time = run.get("start_date_local") 
        if not start_latlng or not start_time:
            continue

        lat, lon = start_latlng
        print(f"Fetching weather for {run['name']} ({start_time})...")
        w = fetch_weather(lat, lon, start_time)

        weather_data.append({
            "id": run["id"],
            "name": run["name"],
            "start_time": start_time,
            "lat": lat,
            "lon": lon,
            "distance_m": run.get("distance"),
            "avg_speed": run.get("average_speed"),
            "avg_hr": run.get("average_heartrate"),
            "avg_cadence": run.get("average_cadence"),
            "weather": w,
        })

    OUT_PATH.write_text(json.dumps(weather_data, indent=2))
    print(f"Saved weather data for {len(weather_data)} runs to {OUT_PATH}")


if __name__ == "__main__":
    print("Running fetch_weather demo...")
    fetch_weather_for_all_runs()
