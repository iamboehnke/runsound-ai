"""
ML-powered music recommender that predicts optimal playlist features
based on planned run characteristics.
"""
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
import random

from spotify_client import SpotifyClient, SpotifyAuthError

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "music_recommender_model.pkl"
OUT_PLAYLIST_METADATA_PATH = DATA_DIR / "latest_playlist.json"

# Import from training module
try:
    from train_music_model import predict_music_features
except ImportError:
    predict_music_features = None


def search_tracks_by_query(sp: SpotifyClient, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search for tracks using a query string."""
    url = "https://api.spotify.com/v1/search"
    params = {'q': query, 'type': 'track', 'limit': limit}
    
    import requests
    r = requests.get(url, headers=sp._headers(), params=params, timeout=15)
    
    if r.status_code != 200:
        print(f"Warning: Search failed for query '{query}': {r.status_code}")
        return []
    
    data = r.json()
    return data.get('tracks', {}).get('items', [])


def get_search_queries_for_run(run_features: Dict[str, Any]) -> List[str]:
    """Generate search queries based on run characteristics, focusing on preferred genres."""
    run_type = run_features.get("run_type", "steady")
    pace = run_features.get("avg_pace_min_km", 6.0)
    
    # Define your preferred genres
    PREFERRED_GENRES = ["pop", "indie", "rap"]
    GENRE_STRING = " OR ".join(PREFERRED_GENRES) # Creates a string like "pop OR indie OR rap"

    # Base queries by run type
    # These queries are now generic to allow the genre string to shape the result
    query_map = {
        "interval": ["high intensity workout", "interval training music", "fast tempo running"],
        "tempo": ["tempo run playlist", "threshold running", "upbeat workout"],
        "easy": ["easy running", "recovery run music", "chill workout"],
        "race": ["race day music", "high energy running", "motivation workout"],
        "long": ["long run playlist", "endurance running", "steady pace music"],
        "steady": ["running music", "jogging playlist", "workout mix"],
    }
    
    # 1. Get the base queries for the run type
    base_queries = query_map.get(run_type, query_map["steady"])
    # 2. Append genre bias to each query
    queries = [f"{q} ({GENRE_STRING})" for q in base_queries]
    # 3. Add pace-based queries (also with genre bias)
    if pace < 5.0:
        queries.append(f"fast running music ({GENRE_STRING})")
    elif pace > 6.5:
        queries.append(f"slow jog playlist ({GENRE_STRING})")
    
    return queries


def load_ml_model() -> Dict[str, Any]:
    """Load trained ML model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model not found at {MODEL_PATH}. "
            "Please run train_music_model.py first to train the model."
        )
    
    with open(MODEL_PATH, "rb") as f:
        model_artifacts = pickle.load(f)
    
    print("ML model loaded successfully")
    return model_artifacts


def get_run_features_from_user_input(
    pace_str: str,
    distance_km: float,
    run_type: str,
    temp_c: float = 15.0,
    time_of_day: str = "Morning"
) -> Dict[str, Any]:
    """
    Convert user input into feature dictionary for ML prediction.
    
    Args:
        pace_str: Pace as string (e.g., "5:30" for 5:30 min/km)
        distance_km: Distance in kilometers
        run_type: Type of run (easy, tempo, interval, long, race, steady)
        temp_c: Temperature in Celsius
        time_of_day: Morning, Afternoon, Evening, Night
    """
    # Parse pace
    if ":" in pace_str:
        parts = pace_str.split(":")
        pace_min_km = int(parts[0]) + int(parts[1]) / 60.0
    else:
        pace_min_km = float(pace_str)
    
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
        # Default values for other features
        "precipitation": 0,
        "windspeed_kmh": 0,
        "humidity": 50,
        "elevation_gain_m": 0,
        "pace_consistency": 0.3,
        "weekly_mileage_km": 30,
    }


def generate_playlist_title(run_features: Dict[str, Any], music_features: Dict[str, Any]) -> str:
    """Generate descriptive playlist title."""
    run_type = run_features.get("run_type", "").title()
    pace = run_features.get("avg_pace_min_km", 0)
    distance = run_features.get("distance_km", 0)
    
    # Format pace
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    pace_str = f"{minutes}:{seconds:02d}"
    
    tempo = music_features.get("target_tempo", 150)
    
    return f"RunSound AI - {run_type} Run | {pace_str}/km | {distance:.1f}km @ {tempo} BPM"


def recommend_and_create_playlist_ml():
    """ML-powered playlist generation."""
    
    print("\n" + "="*60)
    print("ML-POWERED RUNSOUND AI RECOMMENDER")
    print("="*60)
    
    # Get user input for planned run
    print("\n--- Plan Your Run ---")
    
    # Simple CLI input (can be replaced with GUI later)
    pace_input = input("Target pace (e.g., '5:30' for 5:30 min/km): ").strip()
    distance_input = input("Distance (km): ").strip()
    
    print("\nRun types: easy, tempo, interval, long, race, steady")
    run_type_input = input("Run type: ").strip().lower()
    
    temp_input = input("Expected temperature (°C, default 15): ").strip()
    temp_c = float(temp_input) if temp_input else 15.0
    
    print("\nTime of day: Morning, Afternoon, Evening, Night")
    time_input = input("Time (default Morning): ").strip()
    time_of_day = time_input if time_input else "Morning"
    
    # Convert inputs to features
    try:
        run_features = get_run_features_from_user_input(
            pace_str=pace_input,
            distance_km=float(distance_input),
            run_type=run_type_input if run_type_input else "steady",
            temp_c=temp_c,
            time_of_day=time_of_day
        )
    except Exception as e:
        print(f"\nError parsing input: {e}")
        return
    
    print("\n--- Run Summary ---")
    print(f"  Pace: {run_features['avg_pace_min_km']:.2f} min/km")
    print(f"  Distance: {run_features['distance_km']:.1f} km")
    print(f"  Type: {run_features['run_type']}")
    print(f"  Temperature: {run_features['temp_c']:.1f}°C")
    print(f"  Time: {run_features['time_of_day']}")
    
    try:
        # Load ML model
        model_artifacts = load_ml_model()
        
        # Predict music features
        print("\n--- ML Prediction ---")
        music_features = predict_music_features(model_artifacts, run_features)
        
        print(f"  Target Tempo: {music_features['target_tempo']} BPM")
        print(f"  Target Energy: {music_features['target_energy']}")
        print(f"  Target Valence: {music_features['target_valence']}")
        
        # Initialize Spotify
        sp = SpotifyClient()
        
        # Search for tracks
        print("\n--- Searching for Tracks ---")
        queries = get_search_queries_for_run(run_features)
        
        all_tracks = []
        for query in queries:
            print(f"  Searching: '{query}'...")
            tracks = search_tracks_by_query(sp, query, limit=50)
            all_tracks.extend(tracks)
        
        # Remove duplicates
        unique_tracks = {t['id']: t for t in all_tracks if t.get('id')}
        all_tracks = list(unique_tracks.values())
        
        print(f"\n  Found {len(all_tracks)} unique tracks")
        
        if len(all_tracks) < 15:
            print("  Warning: Not many tracks found. Adding generic running music...")
            generic_tracks = search_tracks_by_query(sp, "running music", limit=50)
            all_tracks.extend(generic_tracks)
            unique_tracks = {t['id']: t for t in all_tracks if t.get('id')}
            all_tracks = list(unique_tracks.values())
        
        # Select 30 random tracks
        selected_tracks = random.sample(all_tracks, min(30, len(all_tracks)))
        track_uris = [t['uri'] for t in selected_tracks]
        
        # Create playlist
        playlist_title = generate_playlist_title(run_features, music_features)
        playlist_desc = (
            f"ML-generated playlist for your {run_features['run_type']} run. "
            f"Predicted targets: {music_features['target_tempo']} BPM, "
            f"Energy: {music_features['target_energy']}, "
            f"Valence: {music_features['target_valence']}"
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
        
        # Save metadata
        output_metadata = {
            "run_features": run_features,
            "predicted_music_features": music_features,
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "title": playlist_title,
            "track_count": len(track_uris),
        }
        
        OUT_PLAYLIST_METADATA_PATH.write_text(json.dumps(output_metadata, indent=2))
        
        print("\n" + "="*60)
        print("SUCCESS! ML-Powered Playlist Generated")
        print("="*60)
        print(f"Playlist URL: {playlist_url}")
        print(f"Tracks: {len(track_uris)}")
        print("="*60)
        
        # Auto-open in browser
        import webbrowser
        webbrowser.open(playlist_url)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except SpotifyAuthError as e:
        print(f"\nSpotify Auth Error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    recommend_and_create_playlist_ml()