"""
feature_engineer.py
Merges the Strava activity data with the OpenWeatherMap data.
Calculates the derived metrics (features) needed for the recommender engine:
- Avg Pace (min/km, converted to target BPM range)
- Time of Day (e.g., 'Morning', 'Afternoon', 'Night')
- Temperature Bin ('Cold', 'Mild', 'Warm')
- Run Length Bin ('Short', 'Medium', 'Long')
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEATHER_PATH = DATA_DIR / "run_weather.json"
OUT_PATH = DATA_DIR / "featured_runs.json"

# --- Constants for Feature Mapping ---
PACE_TO_BPM_MAPPING = {
    # Pace in min/km: Target BPM
    (4.5, 5.0): 170, # Fast
    (5.0, 5.5): 165, # Medium-Fast
    (5.5, 6.0): 160, # Medium
    (6.0, 7.0): 150, # Slow/Jog
}

TEMP_BINS = {
    "Cold": (-100, 5),    # < 5°C
    "Mild": (5, 18),      # 5°C to 18°C
    "Warm": (18, 100),    # > 18°C
}

# --- Helper Functions (Borrowed from fetch_strava.py for convenience) ---
def avg_pace_min_per_km(activity_json: Dict[str, Any]) -> float:
    """Strava average_speed is in m/s. Convert to min/km (float)."""
    # Assuming the data format from fetch_weather.py which includes 'avg_speed'
    avg_speed = activity_json.get("avg_speed")  # m/s from fetch_weather.py output
    if avg_speed and avg_speed > 0:
        return 1000.0 / avg_speed / 60.0
    return 0.0 # Return 0 or handle error


# --- Feature Engineering Functions ---

def map_pace_to_bpm(pace_min_km: float) -> int:
    """Maps pace (min/km) to a single target BPM."""
    # Find the corresponding BPM based on the pace
    for (lower, upper), bpm in PACE_TO_BPM_MAPPING.items():
        if lower <= pace_min_km < upper:
            return bpm
    
    # Default/fallback BPM for paces outside the map (e.g., very slow or very fast)
    if pace_min_km < min(k[0] for k in PACE_TO_BPM_MAPPING.keys()):
        return 175
    return 140 # Default to a lower BPM for very slow runs

def get_time_of_day(timestamp: str) -> str:
    """Extracts Time of Day feature from the run's start time."""
    # The 'start_time' in weather data is in UTC format (e.g., 2025-10-22T06:00:00Z)
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    hour = dt.hour
    
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def get_temp_bin(temp_celsius: float) -> str:
    """Bins the temperature into Cold, Mild, or Warm."""
    for bin_name, (lower, upper) in TEMP_BINS.items():
        if lower <= temp_celsius < upper:
            return bin_name
    return "Other" # Should not happen with current bins

def get_run_length_bin(distance_m: float) -> str:
    """Bins the run distance into Short, Medium, or Long."""
    distance_km = distance_m / 1000.0
    if distance_km < 5:
        return "Short"
    elif distance_km < 10:
        return "Medium"
    else:
        return "Long"

def feature_engineer_runs() -> List[Dict[str, Any]]:
    """Loads run and weather data, engineers features, and saves the result."""
    if not WEATHER_PATH.exists():
        raise FileNotFoundError("run_weather.json not found. Run fetch_weather.py first.")
    
    runs_with_weather = json.loads(WEATHER_PATH.read_text())
    featured_runs = []

    for run in runs_with_weather:
        # 1. Core Metrics
        pace_min_km = avg_pace_min_per_km(run)
        
        # Check for missing weather data gracefully
        weather = run.get("weather", {})
        temp_c = weather.get("temperature_2m", math.nan)
        
        # 2. Derived Features
        target_bpm = map_pace_to_bpm(pace_min_km)
        time_of_day = get_time_of_day(run["start_time"])
        temp_bin = get_temp_bin(temp_c)
        run_length_bin = get_run_length_bin(run.get("distance_m", 0))

        # 3. Create Feature Dictionary
        features = {
            "run_id": run["id"],
            "name": run["name"],
            "start_time_utc": run["start_time"],
            "distance_km": run.get("distance_m", 0) / 1000.0,
            "avg_pace_min_km": round(pace_min_km, 2),
            "avg_hr": run.get("avg_hr"),
            "temp_c": temp_c,
            "precipitation": weather.get("precipitation"),
            "windspeed_kmh": weather.get("windspeed_10m"),
            
            # --- THE KEY FEATURES FOR RECOMMENDATION ---
            "target_bpm": target_bpm,
            "run_length_bin": run_length_bin,
            "temp_bin": temp_bin,
            "time_of_day": time_of_day,
        }
        
        featured_runs.append(features)

    # Save the result
    OUT_PATH.write_text(json.dumps(featured_runs, indent=2))
    print(f"Engineered features for {len(featured_runs)} runs — saved to {OUT_PATH}")
    return featured_runs


if __name__ == "__main__":
    print("Running feature_engineer demo...")
    try:
        data = feature_engineer_runs()
        if data:
            print("\n--- Latest Run Features ---")
            print(json.dumps(data[0], indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")