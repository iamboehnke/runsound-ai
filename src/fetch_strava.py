"""
fetch_strava.py
Utilities to authenticate with Strava, refresh access token, fetch latest run activity,
and cache responses for reproducibility.

Expected .env keys (do NOT commit real .env to git):
  STRAVA_CLIENT_ID
  STRAVA_CLIENT_SECRET
  STRAVA_REFRESH_TOKEN
  STRAVA_API_BASE (optional)  -> defaults to https://www.strava.com
  DATA_DIR (optional)         -> defaults to ../data

Usage:
  python src/fetch_strava.py   # will print the latest run and save to data/latest_run.json
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

# Load .env from project root (adjust if your .env is somewhere else)
load_dotenv()

# Config from environment
STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
STRAVA_REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")
STRAVA_API_BASE = os.getenv("STRAVA_API_BASE", "https://www.strava.com")
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = DATA_DIR / "latest_runs.json"
TOKEN_CACHE = DATA_DIR / "strava_token_cache.json"


class StravaAuthError(RuntimeError):
    pass


def refresh_strava_token(refresh_token: str = STRAVA_REFRESH_TOKEN,
                        client_id: str = STRAVA_CLIENT_ID,
                        client_secret: str = STRAVA_CLIENT_SECRET,
                        use_token_cache: bool = True) -> Dict[str, Any]:
    """
    Exchange a refresh token for a new access token (and expiry).
    Caches token response (access_token + expires_at) to avoid unnecessary refreshes.

    Returns token_json with keys: access_token, expires_at, refresh_token, athlete, ...
    """
    if not refresh_token or not client_id or not client_secret:
        raise StravaAuthError("Missing STRAVA_CLIENT_ID/STRAVA_CLIENT_SECRET/STRAVA_REFRESH_TOKEN in .env")

    # Try cached token (if present and not expired)
    if use_token_cache and TOKEN_CACHE.exists():
        try:
            cached = json.loads(TOKEN_CACHE.read_text())
            expires_at = cached.get("expires_at", 0)
            now = int(time.time())
            # keep if > 60s left
            if expires_at - now > 60:
                return cached
        except Exception:
            # ignore cache read errors and continue to refresh
            pass

    url = f"{STRAVA_API_BASE}/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    resp = requests.post(url, data=payload, timeout=15)
    if resp.status_code != 200:
        raise StravaAuthError(f"Failed to refresh token: {resp.status_code} {resp.text}")

    token_json = resp.json()
    # Cache token response
    try:
        TOKEN_CACHE.write_text(json.dumps(token_json))
    except Exception:
        pass

    return token_json


def get_access_token() -> str:
    token_info = refresh_strava_token()
    access_token = token_info.get("access_token")
    if not access_token:
        raise StravaAuthError("No access_token returned from Strava.")
    return access_token


def get_latest_runs(max_runs: int = 50, access_token: Optional[str] = None) -> list[Dict[str, Any]]:
    """Fetch up to `max_runs` recent 'Run' activities with coordinates."""
    if access_token is None:
        access_token = get_access_token()

    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{STRAVA_API_BASE}/api/v3/athlete/activities"

    runs = []
    page = 1
    per_page = 30

    while len(runs) < max_runs:
        params = {"page": page, "per_page": per_page}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch activities: {resp.status_code} {resp.text}")

        activities = resp.json()
        if not activities:
            break

        page_runs = [a for a in activities if a.get("type") == "Run" and a.get("start_latlng")]
        runs.extend(page_runs)

        if len(runs) >= max_runs:
            runs = runs[:max_runs]
            break

        page += 1

    CACHE_PATH.write_text(json.dumps(runs, indent=2))
    print(f"Fetched {len(runs)} runs — saved to {CACHE_PATH}")
    return runs


def get_activity_streams(activity_id: int, keys: str = "time,latlng,altitude,heartrate,velocity_smooth,cadence") -> Dict[str, Any]:
    """
    Fetch streams (detailed time series) for a given activity.
    keys: comma-separated Strava stream types you want.
    Beware of rate limits and large downloads.
    """
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{STRAVA_API_BASE}/api/v3/activities/{activity_id}/streams"
    params = {"keys": keys, "key_by_type": "true"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch streams for activity {activity_id}: {resp.status_code} {resp.text}")
    return resp.json()


def load_cached_latest_run() -> Optional[Dict[str, Any]]:
    """Return cached latest run if exists."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return None
    return None


# Small convenience helper: compute avg pace min/km from activity JSON
def avg_pace_min_per_km(activity_json: Dict[str, Any]) -> Optional[float]:
    """
    Strava average_speed is in m/s. Convert to min/km (float).
    Returns minutes per km (e.g., 5.25 -> 5 min 15 sec ~ 5.25).
    """
    avg_speed = activity_json.get("average_speed")  # m/s
    if avg_speed and avg_speed > 0:
        m_per_km = 1000.0
        sec_per_km = m_per_km / avg_speed
        minutes = sec_per_km / 60.0
        return minutes
    # fallback: compute from distance/time if available
    distance = activity_json.get("distance")  # meters
    elapsed = activity_json.get("elapsed_time")  # seconds
    if distance and elapsed and distance > 0:
        sec_per_km = (elapsed / (distance / 1000.0))
        return sec_per_km / 60.0
    return None


if __name__ == "__main__":
    print("Running fetch_strava demo...")
    try:
        # get_latest_runs returns a list of runs
        runs = get_latest_runs() # Renamed 'run' to 'runs' for clarity
        
        if not runs:
            print("No recent Run activity found in the last page of activities.")
        else:
            latest_runs = runs[0] 
            
            print("=== Latest Run Summary ===")
            print(f"id: {latest_runs.get('id')}")
            print(f"name: {latest_runs.get('name')}")
            print(f"start_date_local: {latest_runs.get('start_date_local')}")
            print(f"distance_m: {latest_runs.get('distance')}")
            
            # Use latest_run instead of run for the pace calculation
            pace = avg_pace_min_per_km(latest_runs) 
            
            if pace:
                minutes = int(pace)
                seconds = int(round((pace - minutes) * 60))
                print(f"avg_pace: {minutes}:{seconds:02d} min/km")
            else:
                print("avg_pace: unavailable")
            
            print(f"average_heartrate: {latest_runs.get('average_heartrate')}")
            print(f"average_cadence: {latest_runs.get('average_cadence')}")
            print(f"map_summary_polyline: {'present' if latest_runs.get('map') and latest_runs['map'].get('summary_polyline') else 'none'}")
            print(f"Cached to: {CACHE_PATH}")

            streams = get_activity_streams(latest_runs['id']) 
            print("Streams keys:", streams.keys())

    except Exception as e:
        print("Error:", e)
        raise