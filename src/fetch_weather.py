"""
fetch_weather.py
Fetches weather data (current or historical) for a given run
based on coordinates and datetime.

Expected .env key:
  OPENWEATHER_API_KEY

Optional:
  DATA_DIR (defaults to ../data)

Usage:
  from fetch_strava import load_cached_latest_run
  from fetch_weather import fetch_weather_for_run

  run = load_cached_latest_run()
  weather = fetch_weather_for_run(run)
  print(weather)
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = DATA_DIR / "latest_weather.json"


class WeatherAPIError(RuntimeError):
    pass


def _fetch_onecall(lat: float, lon: float, dt_unix: int) -> Dict[str, Any]:
    """
    Internal helper: fetch historical weather (past 5 days) using onecall/timemachine.
    """
    if not OPENWEATHER_API_KEY:
        raise WeatherAPIError("Missing OPENWEATHER_API_KEY in .env")

    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={dt_unix}&units=metric&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise WeatherAPIError(f"OpenWeather error: {resp.status_code} {resp.text}")
    return resp.json()


def _fetch_current(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fallback for runs older than 5 days (use current conditions).
    """
    if not OPENWEATHER_API_KEY:
        raise WeatherAPIError("Missing OPENWEATHER_API_KEY in .env")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise WeatherAPIError(f"OpenWeather error: {resp.status_code} {resp.text}")
    return resp.json()


def extract_run_coords_and_time(run_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to extract coordinates and start time from Strava activity.
    """
    try:
        latlng = run_json.get("start_latlng")
        if not latlng or len(latlng) != 2:
            # fallback: use end_latlng
            latlng = run_json.get("end_latlng")
        if not latlng or len(latlng) != 2:
            return None

        start_date_local = run_json.get("start_date_local")
        if not start_date_local:
            start_date_local = run_json.get("start_date")

        # parse to unix timestamp
        dt = datetime.fromisoformat(start_date_local.replace("Z", "+00:00"))
        dt_unix = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return {"lat": latlng[0], "lon": latlng[1], "timestamp": dt_unix}
    except Exception:
        return None


def fetch_weather_for_run(run_json: Dict[str, Any], force_current: bool = False) -> Dict[str, Any]:
    """
    Given a Strava run JSON, fetch the most relevant weather (historical if recent enough).
    Saves to data/latest_weather.json
    """
    info = extract_run_coords_and_time(run_json)
    if not info:
        raise ValueError("Run JSON missing coordinates or time.")

    lat, lon, ts = info["lat"], info["lon"], info["timestamp"]
    now = int(time.time())
    five_days_sec = 5 * 24 * 3600

    if force_current or (now - ts > five_days_sec):
        raw = _fetch_current(lat, lon)
        mode = "current"
    else:
        raw = _fetch_onecall(lat, lon, ts)
        mode = "historical"

    # Compact summary for downstream use
    summary = _extract_summary(raw, mode)
    summary["mode"] = mode
    summary["lat"] = lat
    summary["lon"] = lon
    summary["timestamp"] = ts

    try:
        CACHE_PATH.write_text(json.dumps(summary, indent=2))
    except Exception:
        pass

    return summary


def _extract_summary(raw: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Extracts a compact weather summary.
    """
    try:
        if mode == "historical":
            current = raw.get("data", [])[0] if "data" in raw else raw.get("hourly", [{}])[0]
        else:
            current = raw.get("current", raw)
    except Exception:
        current = raw

    weather_desc = ""
    if "weather" in current and isinstance(current["weather"], list) and current["weather"]:
        weather_desc = current["weather"][0].get("description", "")

    return {
        "temp": current.get("temp"),
        "feels_like": current.get("feels_like"),
        "humidity": current.get("humidity"),
        "wind_speed": current.get("wind_speed"),
        "clouds": current.get("clouds"),
        "weather_desc": weather_desc,
        "pressure": current.get("pressure"),
    }


def load_cached_latest_weather() -> Optional[Dict[str, Any]]:
    """Return cached weather if exists."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return None
    return None


if __name__ == "__main__":
    # Demo: fetch weather for cached run
    from fetch_strava import load_cached_latest_run

    print("Running fetch_weather demo...")
    run = load_cached_latest_run()
    if not run:
        print("No cached run found. Run fetch_strava.py first.")
    else:
        w = fetch_weather_for_run(run)
        print(json.dumps(w, indent=2))
        print(f"Cached to: {CACHE_PATH}")
