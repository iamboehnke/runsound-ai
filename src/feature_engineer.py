"""
The main feature engineering script for RunSound AI.
Merges Strava activity data with OpenWeatherMap data and calculates
an extensive set of derived, contextual, and historical features
needed for both the heuristic recommender and future ML model training.
"""
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List
import statistics

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEATHER_PATH = DATA_DIR / "run_weather.json"
STRAVA_RUNS_PATH = DATA_DIR / "latest_runs.json" 
OUT_PATH = DATA_DIR / "ml_featured_runs.json"

# --- Constants for Feature Mapping ---
# PACE_TO_BPM_MAPPING is updated to the enhanced version
PACE_TO_BPM_MAPPING = {
    # Pace in min/km: Target BPM
    (4.0, 4.5): 180,  # Very Fast
    (4.5, 5.0): 170,  # Fast
    (5.0, 5.5): 165,  # Medium-Fast
    (5.5, 6.0): 160,  # Medium
    (6.0, 7.0): 150,  # Slow/Jog
    (7.0, 8.0): 140,  # Very Slow
}

# TEMP_BINS is updated to the enhanced version
TEMP_BINS = {
    "Very Cold": (-100, 0),
    "Cold": (0, 10),
    "Mild": (10, 20),
    "Warm": (20, 30),
    "Hot": (30, 100),
}

# --- Helper Functions (Merged and Enhanced) ---

def avg_pace_min_per_km(activity_json: Dict[str, Any]) -> float:
    """Calculate average pace in min/km from speed (m/s)."""
    avg_speed = activity_json.get("avg_speed")
    if avg_speed and avg_speed > 0:
        return 1000.0 / avg_speed / 60.0
    return 0.0

def calculate_elevation_gain(run_json: Dict[str, Any]) -> float:
    """Calculate total elevation gain from Strava data."""
    return run_json.get("total_elevation_gain", 0.0)

def get_pace_consistency(runs_history: List[Dict], current_run_index: int, window: int = 5) -> float:
    """
    Calculate pace consistency (standard deviation of recent paces).
    A lower value indicates more consistent running in the past `window` runs.
    """
    # Get the last `window` runs *before* the current run
    # Note: runs_history is typically ordered chronologically *backwards* by the API fetch
    # We assume runs_history[0] is the LATEST run.
    # To get historical runs relative to current run (runs_history[current_run_index]),
    # we look at indices AFTER the current one in the list.
    
    # We need to adjust the logic based on how the list is ordered.
    # Assuming runs_with_weather is ordered: [Latest Run, Second Latest, ...]
    
    # If runs_history[current_run_index] is the run we are processing,
    # the historical runs are runs_history[current_run_index+1] onwards.
    
    history_slice = runs_history[current_run_index + 1 : current_run_index + 1 + window]
    
    recent_paces = [avg_pace_min_per_km(r) for r in history_slice]
    recent_paces = [p for p in recent_paces if p > 0]
    
    if len(recent_paces) < 2:
        return 0.0
    
    return statistics.stdev(recent_paces)

def get_weekly_mileage(runs_history: List[Dict], current_date: str) -> float:
    """Calculate total distance run in the past 7 days up to the current run."""
    current_dt = datetime.fromisoformat(current_date.replace("Z", "+00:00")).astimezone(timezone.utc)
    week_ago = current_dt - timedelta(days=7)
    
    weekly_distance = 0.0
    for run in runs_history:
        run_date_str = run.get("start_time")
        if not run_date_str:
            continue
            
        run_date = datetime.fromisoformat(run_date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        
        # Only count runs that are within the 7-day window BEFORE or AT the current run's time
        if week_ago <= run_date < current_dt:
            weekly_distance += run.get("distance_m", 0) / 1000.0
            
    return weekly_distance

# --- Feature Engineering Functions (Merged and Enhanced) ---

def map_pace_to_bpm(pace_min_km: float) -> int:
    """Maps pace (min/km) to a single target BPM."""
    for (lower, upper), bpm in PACE_TO_BPM_MAPPING.items():
        if lower <= pace_min_km < upper:
            return bpm
    
    # Fallback based on map limits
    if pace_min_km < min(k[0] for k in PACE_TO_BPM_MAPPING.keys()):
        return 185 # Very fast pace
    return 135 # Very slow pace

def get_time_of_day(timestamp: str) -> str:
    """Extracts Time of Day feature from the run's start time."""
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
    if math.isnan(temp_celsius):
        return "Unknown"
        
    for bin_name, (lower, upper) in TEMP_BINS.items():
        if lower <= temp_celsius < upper:
            return bin_name
    return "Unknown"

def get_run_length_bin(distance_m: float) -> str:
    """Bins the run distance into Short, Medium, Long, or Very Long."""
    distance_km = distance_m / 1000.0
    if distance_km < 5:
        return "Short"
    elif distance_km < 10:
        return "Medium"
    elif distance_km < 15:
        return "Long"
    else:
        return "Very Long"

def detect_run_type(run_name: str, pace_min_km: float, distance_km: float) -> str:
    """
    Detect run type from name and stats.
    Types: easy, tempo, interval, long, race, steady
    """
    name_lower = run_name.lower()
    
    # Check name keywords first
    if any(word in name_lower for word in ["interval", "mas", "repeat", "400m", "200m"]):
        return "interval"
    elif any(word in name_lower for word in ["tempo", "threshold", "marathon pace"]):
        return "tempo"
    elif any(word in name_lower for word in ["easy", "recovery", "slow", "jog"]):
        return "easy"
    elif any(word in name_lower for word in ["race", "competition", "pr"]):
        return "race"
    elif distance_km > 15:
        return "long"
    
    # Infer from pace and distance
    if pace_min_km < 4.5:
        return "race"
    elif pace_min_km < 5.0:
        return "tempo"
    elif distance_km > 15:
        return "long"
    elif pace_min_km > 6.5:
        return "easy"
    else:
        return "steady"

def calculate_music_targets(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate target music features (Energy, Valence) based on run characteristics.
    These features will serve as ML labels for training.
    """
    # Base targets
    target_tempo = features["target_bpm"]
    target_energy = 0.6
    target_valence = 0.5
    
    # Adjust by run type
    run_type = features["run_type"]
    if run_type == "interval":
        target_energy = 0.85
        target_valence = 0.7
    elif run_type == "tempo":
        target_energy = 0.75
        target_valence = 0.65
    elif run_type == "easy":
        target_energy = 0.4
        target_valence = 0.6
    elif run_type == "race":
        target_energy = 0.9
        target_valence = 0.8
    elif run_type == "long":
        target_energy = 0.55
        target_valence = 0.6
    
    # Adjust by temperature
    temp_bin = features["temp_bin"]
    if temp_bin in ["Very Cold", "Cold"]:
        target_valence = max(0.2, target_valence - 0.15)
    elif temp_bin in ["Warm", "Hot"]:
        target_valence = min(0.9, target_valence + 0.15)
    
    # Adjust by time of day
    time_of_day = features["time_of_day"]
    if time_of_day == "Morning":
        target_valence = min(0.8, target_valence + 0.1)
    elif time_of_day == "Night":
        target_energy = max(0.3, target_energy - 0.15)
    
    return {
        "target_tempo": target_tempo, # Redundant, but explicit for targets
        "target_energy": round(target_energy, 2),
        "target_valence": round(target_valence, 2),
    }

def feature_engineer_runs() -> List[Dict[str, Any]]:
    """
    Main function: Load data and engineer the full set of ML-ready features.
    """
    if not WEATHER_PATH.exists():
        raise FileNotFoundError("run_weather.json not found. Run fetch_weather.py first.")
    
    if not STRAVA_RUNS_PATH.exists():
        # NOTE: This file is only needed for elevation_gain. We can proceed without it
        # but elevation_gain will be 0.0 for all runs.
        print(f"Warning: {STRAVA_RUNS_PATH.name} not found. Elevation gain will be 0.0.")
        strava_runs = []
    else:
        strava_runs = json.loads(STRAVA_RUNS_PATH.read_text())
        
    runs_with_weather = json.loads(WEATHER_PATH.read_text())
    
    # Create a lookup for Strava runs by ID
    strava_lookup = {r.get("id"): r for r in strava_runs if r.get("id")}
    
    featured_runs = []

    # Iterate through runs (assuming runs_with_weather is sorted from newest to oldest)
    for idx, run in enumerate(runs_with_weather):
        run_id = run.get("id")
        strava_run = strava_lookup.get(run_id, {})
        
        # 1. Core Metrics
        pace_min_km = avg_pace_min_per_km(run)
        distance_km = run.get("distance_m", 0) / 1000.0
        weather = run.get("weather", {})
        temp_c = weather.get("temperature_2m", math.nan)
        
        # 2. Enhanced Features
        elevation_gain = calculate_elevation_gain(strava_run)
        
        # Historical/Contextual Features
        # The history slice needs to be computed based on how runs_with_weather is ordered
        pace_consistency = get_pace_consistency(runs_with_weather, idx, window=5)
        weekly_mileage = get_weekly_mileage(runs_with_weather, run["start_time"])
        
        # 3. Derived Features
        target_bpm = map_pace_to_bpm(pace_min_km)
        time_of_day = get_time_of_day(run["start_time"])
        temp_bin = get_temp_bin(temp_c)
        run_length_bin = get_run_length_bin(run.get("distance_m", 0))
        run_type = detect_run_type(run["name"], pace_min_km, distance_km)

        # 4. Create Feature Dictionary
        features = {
            # Identifiers
            "run_id": run_id,
            "name": run["name"],
            "start_time_utc": run["start_time"],
            
            # Basic Stats
            "distance_km": round(distance_km, 2),
            "avg_pace_min_km": round(pace_min_km, 2),
            "avg_hr": run.get("avg_hr"),
            "elevation_gain_m": round(elevation_gain, 1), # Enhanced
            
            # Weather
            "temp_c": temp_c,
            "temp_bin": temp_bin,
            "precipitation": weather.get("precipitation"),
            "windspeed_kmh": weather.get("windspeed_10m"),
            "humidity": weather.get("relative_humidity_2m"), # Enhanced
            "feels_like_c": weather.get("apparent_temperature"), # Enhanced
            
            # Contextual
            "time_of_day": time_of_day,
            "run_length_bin": run_length_bin,
            "run_type": run_type, # Enhanced
            
            # Training Load Features
            "pace_consistency": round(pace_consistency, 2), # Enhanced
            "weekly_mileage_km": round(weekly_mileage, 1), # Enhanced
            
            # Music Targets (Input for Recommender/ML)
            "target_bpm": target_bpm,
        }
        
        # 5. Add Heuristic Music Targets (ML Labels)
        music_targets = calculate_music_targets(features)
        features.update(music_targets)
        
        featured_runs.append(features)

    # Save the result
    OUT_PATH.write_text(json.dumps(featured_runs, indent=2))
    print(f"Engineered ML features for {len(featured_runs)} runs")
    print(f"Saved to: {OUT_PATH}")
    return featured_runs


if __name__ == "__main__":
    print("Running enhanced feature engineering for ML...")
    try:
        data = feature_engineer_runs()
        if data:
            print("\n--- Sample Run Features (Latest) ---")
            print(json.dumps(data[0], indent=2))
            
            # Show distribution of run types
            run_types = {}
            for run in data:
                rt = run["run_type"]
                run_types[rt] = run_types.get(rt, 0) + 1
            
            print("\n--- Run Type Distribution ---")
            for rt, count in sorted(run_types.items(), key=lambda x: -x[1]):
                print(f"  {rt}: {count}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()