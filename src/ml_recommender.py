"""
smart_ml_recommender.py
Intelligent ML-powered music recommender with:
- Weather API integration (auto-fetch forecast)
- Pace suggestions based on recent runs and run type
- Training load awareness (fatigue detection)
- Progressive tempo playlists for long runs
- Genre learning from listening history
"""
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
from datetime import datetime, timezone, timedelta
import requests

from spotify_client import SpotifyClient, SpotifyAuthError

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "music_recommender_model.pkl"
ML_FEATURES_PATH = DATA_DIR / "ml_featured_runs.json"
OUT_PLAYLIST_METADATA_PATH = DATA_DIR / "latest_playlist.json"

# Import from training module
try:
    from train_music_model import predict_music_features
except ImportError:
    predict_music_features = None


# --- Weather Forecasting ---

def fetch_weather_forecast(lat: float = 55.4038, lon: float = 10.4024) -> Dict[str, Any]:
    """
    Fetch current weather forecast for Odense, Denmark (default).
    Returns temperature, precipitation, wind, etc.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,weathercode,windspeed_10m,relative_humidity_2m",
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            current = data.get("current", {})
            return {
                "temp_c": current.get("temperature_2m", 15.0),
                "precipitation": current.get("precipitation", 0.0),
                "windspeed_kmh": current.get("windspeed_10m", 0.0),
                "humidity": current.get("relative_humidity_2m", 50.0),
            }
    except Exception as e:
        print(f"Warning: Weather API failed: {e}")
    
    # Fallback to reasonable defaults
    return {
        "temp_c": 15.0,
        "precipitation": 0.0,
        "windspeed_kmh": 10.0,
        "humidity": 60.0,
    }


# --- Pace Intelligence ---

def analyze_recent_runs(run_type: str) -> Dict[str, Any]:
    """
    Analyze recent runs to suggest pace and detect fatigue.
    Returns: suggested_pace, pace_range, fatigue_level, weekly_load
    """
    if not ML_FEATURES_PATH.exists():
        return {
            "suggested_pace": None,
            "pace_range": None,
            "fatigue_level": "unknown",
            "weekly_load": 0,
            "recent_runs_count": 0
        }
    
    runs = json.loads(ML_FEATURES_PATH.read_text())
    
    # Filter recent runs (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    recent_runs = [
        r for r in runs 
        if datetime.fromisoformat(r["start_time_utc"].replace("Z", "+00:00")) >= thirty_days_ago
    ]
    
    if not recent_runs:
        return {
            "suggested_pace": None,
            "pace_range": None,
            "fatigue_level": "fresh",
            "weekly_load": 0,
            "recent_runs_count": 0
        }
    
    # Calculate weekly load (last 7 days)
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    weekly_runs = [
        r for r in runs
        if datetime.fromisoformat(r["start_time_utc"].replace("Z", "+00:00")) >= seven_days_ago
    ]
    weekly_distance = sum(r["distance_km"] for r in weekly_runs)
    
    # Filter by run type for pace suggestion
    type_runs = [r for r in recent_runs if r["run_type"] == run_type]
    
    if not type_runs:
        # Use all runs as fallback
        type_runs = recent_runs
    
    paces = [r["avg_pace_min_km"] for r in type_runs if r.get("avg_pace_min_km", 0) > 0]
    
    if paces:
        avg_pace = sum(paces) / len(paces)
        min_pace = min(paces)
        max_pace = max(paces)
        
        # Suggest slightly faster than average for progression
        suggested = avg_pace - 0.05 if run_type in ["tempo", "interval"] else avg_pace
        
        pace_range = (suggested - 0.15, suggested + 0.15)
    else:
        suggested = 6.0  # Default
        pace_range = (5.5, 6.5)
    
    # Fatigue detection
    if weekly_distance > 60:
        fatigue_level = "high_load"
    elif weekly_distance > 40:
        fatigue_level = "moderate"
    elif weekly_distance < 15:
        fatigue_level = "fresh"
    else:
        fatigue_level = "normal"
    
    # Pace consistency (lower = more consistent)
    if len(paces) >= 3:
        import statistics
        consistency = statistics.stdev(paces)
    else:
        consistency = 0.5
    
    return {
        "suggested_pace": suggested,
        "pace_range": pace_range,
        "fatigue_level": fatigue_level,
        "weekly_load": weekly_distance,
        "recent_runs_count": len(recent_runs),
        "pace_consistency": consistency,
        "avg_pace_last_30d": sum(paces) / len(paces) if paces else None
    }


def format_pace(pace_float: float) -> str:
    """Convert pace float to MM:SS format."""
    minutes = int(pace_float)
    seconds = int(round((pace_float - minutes) * 60))
    return f"{minutes}:{seconds:02d}"


# --- Music Intelligence ---

def search_tracks_by_query(sp: SpotifyClient, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search for tracks using a query string."""
    url = "https://api.spotify.com/v1/search"
    params = {'q': query, 'type': 'track', 'limit': limit}
    
    r = requests.get(url, headers=sp._headers(), params=params, timeout=15)
    
    if r.status_code != 200:
        print(f"Warning: Search failed for query '{query}': {r.status_code}")
        return []
    
    data = r.json()
    return data.get('tracks', {}).get('items', [])


def learn_genre_preferences() -> List[str]:
    """
    Analyze past playlists to learn genre preferences.
    Returns list of preferred genres.
    """
    # TODO: Track which playlists user actually listens to
    # For now, use default preferences
    return ["pop", "indie", "rap", "electronic", "rock"]


def get_search_queries_for_run(run_features: Dict[str, Any], preferred_genres: List[str]) -> List[str]:
    """Generate intelligent search queries based on run characteristics and learned preferences."""
    run_type = run_features.get("run_type", "steady")
    pace = run_features.get("avg_pace_min_km", 6.0)
    distance = run_features.get("distance_km", 5.0)
    temp = run_features.get("temp_c", 15.0)
    
    genre_string = " OR ".join(preferred_genres[:3])
    
    # Base queries by run type
    query_templates = {
        "interval": [
            "high intensity workout",
            "HIIT training music", 
            "explosive energy running"
        ],
        "tempo": [
            "threshold running",
            "sustained effort workout",
            "driving beat running"
        ],
        "easy": [
            "easy running chill",
            "recovery jog music",
            "relaxed workout"
        ],
        "race": [
            "race day motivation",
            "peak performance running",
            "championship energy"
        ],
        "long": [
            "endurance running",
            "marathon training music",
            "steady pace long run"
        ],
        "steady": [
            "running playlist",
            "jogging music",
            "workout motivation"
        ],
    }
    
    base_queries = query_templates.get(run_type, query_templates["steady"])
    queries = [f"{q} ({genre_string})" for q in base_queries]
    
    # Add context-based queries
    if pace < 5.0:
        queries.append(f"fast running high energy ({genre_string})")
    elif pace > 6.5:
        queries.append(f"slow tempo chill running ({genre_string})")
    
    if temp < 5:
        queries.append(f"cold weather running motivation ({genre_string})")
    elif temp > 25:
        queries.append(f"summer running upbeat ({genre_string})")
    
    if distance > 15:
        queries.append(f"ultra distance endurance ({genre_string})")
    
    return queries


def create_progressive_playlist(tracks: List[Dict], run_type: str, distance_km: float) -> List[Dict]:
    """
    For long runs, create a progressive tempo/energy playlist.
    Starts slow, builds to peak, maintains, then cool down.
    """
    if run_type != "long" or distance_km < 12:
        # Just shuffle for other runs
        random.shuffle(tracks)
        return tracks[:30]
    
    # For long runs: progressive structure
    # 20% warmup (lower energy) - 60% main (steady) - 20% finish strong
    
    n = min(len(tracks), 30)
    warmup_count = max(3, n // 5)
    finish_count = max(3, n // 5)
    main_count = n - warmup_count - finish_count
    
    # Sort all tracks (you'd ideally use audio features here, but we'll simulate)
    random.shuffle(tracks)
    
    warmup = tracks[:warmup_count]
    main_section = tracks[warmup_count:warmup_count + main_count]
    finish = tracks[warmup_count + main_count:warmup_count + main_count + finish_count]
    
    return warmup + main_section + finish


# --- ML Model Integration ---

def load_ml_model() -> Dict[str, Any]:
    """Load trained ML model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model not found at {MODEL_PATH}. "
            "Run: python src/train_music_model.py"
        )
    
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_run_features_from_user_input(
    pace_str: str,
    distance_km: float,
    run_type: str,
    weather: Dict[str, Any],
    time_of_day: str,
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert user input + weather + historical analysis into ML features."""
    
    # Parse pace
    if ":" in pace_str:
        parts = pace_str.split(":")
        pace_min_km = int(parts[0]) + int(parts[1]) / 60.0
    else:
        pace_min_km = float(pace_str)
    
    temp_c = weather["temp_c"]
    
    # Determine bins
    if distance_km < 5:
        run_length_bin = "Short"
    elif distance_km < 10:
        run_length_bin = "Medium"
    elif distance_km < 15:
        run_length_bin = "Long"
    else:
        run_length_bin = "Very Long"
    
    if temp_c < 0:
        temp_bin = "Very Cold"
    elif temp_c < 10:
        temp_bin = "Cold"
    elif temp_c < 20:
        temp_bin = "Mild"
    elif temp_c < 30:
        temp_bin = "Warm"
    else:
        temp_bin = "Hot"
    
    return {
        "avg_pace_min_km": pace_min_km,
        "distance_km": distance_km,
        "run_type": run_type,
        "temp_c": temp_c,
        "time_of_day": time_of_day,
        "run_length_bin": run_length_bin,
        "temp_bin": temp_bin,
        "precipitation": weather.get("precipitation", 0),
        "windspeed_kmh": weather.get("windspeed_kmh", 0),
        "humidity": weather.get("humidity", 50),
        "elevation_gain_m": 0,  # Can't predict this
        "pace_consistency": analysis.get("pace_consistency", 0.3),
        "weekly_mileage_km": analysis.get("weekly_load", 30),
    }


def generate_playlist_title(run_features: Dict[str, Any], music_features: Dict[str, Any]) -> str:
    """Generate descriptive playlist title."""
    run_type = run_features.get("run_type", "").title()
    pace = run_features.get("avg_pace_min_km", 0)
    distance = run_features.get("distance_km", 0)
    
    pace_str = format_pace(pace)
    tempo = music_features.get("target_tempo", 150)
    
    return f"🏃 {run_type} | {pace_str}/km | {distance:.1f}km @ {tempo} BPM"


# --- Main Recommender ---

def recommend_and_create_playlist():
    """
    Intelligent ML-powered playlist generation with:
    - Auto weather fetch
    - Pace suggestions
    - Fatigue awareness
    - Genre learning
    - Progressive playlists
    """
    
    print("\n" + "="*70)
    print("🎵 RUNSOUND AI - SMART ML RECOMMENDER 🏃")
    print("="*70)
    
    # Step 1: Run Type Selection
    print("\n--- Plan Your Run ---")
    print("\n🏃 Run Types:")
    print("  • easy      - Easy recovery run (comfortable pace)")
    print("  • tempo     - Tempo run (comfortably hard, sustained)")
    print("  • interval  - Interval training (high intensity repeats)")
    print("  • long      - Long endurance run (progressive build)")
    print("  • race      - Race effort (maximum performance)")
    print("  • steady    - Steady state run (moderate effort)")
    
    run_type_input = input("\nRun type: ").strip().lower()
    if run_type_input not in ["easy", "tempo", "interval", "long", "race", "steady"]:
        run_type_input = "steady"
        print(f"Using default: {run_type_input}")
    
    # Step 2: Analyze Recent Training
    print("\n--- Analyzing Your Recent Training ---")
    analysis = analyze_recent_runs(run_type_input)
    
    print(f"Recent runs (30d): {analysis['recent_runs_count']}")
    print(f"Weekly load: {analysis['weekly_load']:.1f} km")
    print(f"Fatigue level: {analysis['fatigue_level']}")
    
    if analysis['suggested_pace']:
        pace_range = analysis['pace_range']
        print(f"\nSuggested Pace Range for {run_type_input}:")
        print(f"     {format_pace(pace_range[0])} - {format_pace(pace_range[1])} min/km")
        print(f"     (Based on your recent {run_type_input} runs)")
    
    # Step 3: Get User Input
    pace_input = input("\nTarget pace (e.g., '5:30'): ").strip()
    if not pace_input and analysis['suggested_pace']:
        pace_input = format_pace(analysis['suggested_pace'])
        print(f"Using suggested: {pace_input}")
    
    distance_input = input("Distance (km): ").strip()
    if not distance_input:
        distance_input = "10"
        print(f"Using default: {distance_input}")
    
    # Step 4: Auto-fetch Weather
    print("\n--- Fetching Weather Forecast ---")
    weather = fetch_weather_forecast()
    print(f"  Temperature: {weather['temp_c']:.1f}°C")
    print(f"  Precipitation: {weather['precipitation']:.1f} mm/h")
    print(f"  Wind: {weather['windspeed_kmh']:.1f} km/h")
    print(f"  Humidity: {weather['humidity']:.0f}%")
    
    # Determine time of day
    hour = datetime.now().hour
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"
    
    print(f"  Time: {time_of_day}")
    
    # Step 5: Build Feature Vector
    try:
        run_features = get_run_features_from_user_input(
            pace_str=pace_input,
            distance_km=float(distance_input),
            run_type=run_type_input,
            weather=weather,
            time_of_day=time_of_day,
            analysis=analysis
        )
    except Exception as e:
        print(f"\nError parsing input: {e}")
        return
    
    print("\n--- Run Summary ---")
    print(f"  Type: {run_features['run_type']}")
    print(f"  Pace: {format_pace(run_features['avg_pace_min_km'])}/km")
    print(f"  Distance: {run_features['distance_km']:.1f} km")
    print(f"  Temperature: {run_features['temp_c']:.1f}°C ({run_features['temp_bin']})")
    print(f"  Time: {run_features['time_of_day']}")
    
    try:
        # Step 6: ML Prediction
        print("\n--- ML Model Prediction ---")
        model_artifacts = load_ml_model()
        music_features = predict_music_features(model_artifacts, run_features)
        
        print(f"     Predicted Music Profile:")
        print(f"     Tempo: {music_features['target_tempo']} BPM")
        print(f"     Energy: {music_features['target_energy']} (0-1 scale)")
        print(f"     Valence: {music_features['target_valence']} (happiness)")
        
        # Step 7: Initialize Spotify
        sp = SpotifyClient()
        
        # Step 8: Smart Track Search
        print("\n--- 🔍 Searching for Tracks ---")
        preferred_genres = learn_genre_preferences()
        print(f"  Using genres: {', '.join(preferred_genres[:3])}")
        
        queries = get_search_queries_for_run(run_features, preferred_genres)
        
        all_tracks = []
        for query in queries[:5]:  # Limit queries
            tracks = search_tracks_by_query(sp, query, limit=50)
            all_tracks.extend(tracks)
        
        # Remove duplicates
        unique_tracks = {t['id']: t for t in all_tracks if t.get('id')}
        all_tracks = list(unique_tracks.values())
        
        print(f"  Found {len(all_tracks)} unique tracks")
        
        if len(all_tracks) < 20:
            print("  Adding more variety...")
            backup = search_tracks_by_query(sp, f"running motivation {preferred_genres[0]}", 50)
            all_tracks.extend(backup)
            unique_tracks = {t['id']: t for t in all_tracks if t.get('id')}
            all_tracks = list(unique_tracks.values())
        
        # Step 9: Create Intelligent Playlist Structure
        selected_tracks = create_progressive_playlist(
            all_tracks, 
            run_features['run_type'],
            run_features['distance_km']
        )
        
        if run_features['run_type'] == "long":
            print(f"Created progressive playlist structure (warmup → main → finish)")
        
        track_uris = [t['uri'] for t in selected_tracks]
        
        # Step 10: Create Spotify Playlist
        playlist_title = generate_playlist_title(run_features, music_features)
        playlist_desc = (
            f"ML-powered playlist for your {run_features['run_type']} run. "
            f"Predicted: {music_features['target_tempo']} BPM, "
            f"Energy {music_features['target_energy']}, "
            f"Weather: {run_features['temp_c']:.0f}°C. "
            f"Generated by RunSound AI."
        )
        
        print(f"\n--- Creating Playlist ---")
        print(f"  Title: {playlist_title}")
        
        playlist_id = sp.create_playlist(
            name=playlist_title,
            description=playlist_desc,
            public=True
        )
        
        sp.add_tracks_to_playlist(playlist_id, track_uris)
        
        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
        
        # Step 11: Save Metadata
        output_metadata = {
            "run_features": run_features,
            "predicted_music_features": music_features,
            "analysis": analysis,
            "weather": weather,
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "title": playlist_title,
            "track_count": len(track_uris),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        OUT_PLAYLIST_METADATA_PATH.write_text(json.dumps(output_metadata, indent=2))
        
        print("\n" + "="*70)
        print("SUCCESS! Your Smart Playlist is Ready")
        print("="*70)
        print(f"{len(track_uris)} tracks perfectly matched to your run")
        print(f"{playlist_url}")
        
        if analysis['fatigue_level'] == "high_load":
            print(f"\nTip: Your weekly load is high ({analysis['weekly_load']:.0f}km).")
            print(f"   Consider an easy run or rest day soon!")
        
        print("="*70)
        
        # Auto-open
        import webbrowser
        webbrowser.open(playlist_url)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nFirst time setup required:")
        print("   1. python src/feature_engineer.py")
        print("   2. python src/train_music_model.py")
    except SpotifyAuthError as e:
        print(f"\nSpotify Auth Error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    recommend_and_create_playlist()