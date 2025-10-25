"""
recommender.py
The core engine for RunSound AI.
1. Reads featured run data.
2. Maps context features (pace, temp_bin, time_of_day) to Spotify audio features (tempo, energy, valence).
3. Uses SpotifyClient to generate track recommendations.
4. Creates and populates a new Spotify playlist.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Import components from our project
from spotify_client import SpotifyClient, SpotifyAuthError
from feature_engineer import avg_pace_min_per_km # For pace display in output

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURED_RUNS_PATH = DATA_DIR / "featured_runs.json"
OUT_PLAYLIST_METADATA_PATH = DATA_DIR / "latest_playlist.json"

# --- Recommendation Rules (The Brains of the Operation) ---
def map_context_to_audio_features(run_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates engineered run features into Spotify's 'target' audio features.
    This is the core heuristic (or where a trained ML model would fit).

    Spotify features range from 0.0 to 1.0 (except tempo):
    - valence: musical positiveness (1.0 = happy, 0.0 = sad)
    - energy: intensity and activity (1.0 = energetic)
    - danceability: suitability for dancing/rhythm (1.0 = easy to dance to)
    - tempo: beats per minute (BPM)
    """
    
    # 1. Base Feature Mapping (Pace is primary driver for Tempo)
    target_bpm = run_features.get("target_bpm", 150)
    target_tempo = target_bpm # Spotify uses 'target_tempo' as a hint
    target_energy = 0.6
    target_valence = 0.5
    
    # 2. Contextual Adjustments
    
    # --- Weather/Temperature (Valence/Energy) ---
    temp_bin = run_features.get("temp_bin")
    if temp_bin == "Cold":
        # Low valence (ambient/chill) but potentially medium-high energy for warmth
        target_valence = 0.35 
        target_energy = 0.65
    elif temp_bin == "Warm":
        # High valence (upbeat/happy)
        target_valence = 0.8
        target_energy = 0.75

    # --- Time of Day (Valence/Energy Shift) ---
    time_of_day = run_features.get("time_of_day")
    if time_of_day == "Morning":
        # Start calm, progressive
        target_energy = min(0.7, target_energy + 0.05) 
        target_valence = min(0.7, target_valence + 0.1) # Boost mood for the start of the day
    elif time_of_day == "Night":
        # Lower energy/valence for winding down or staying focused
        target_energy = max(0.4, target_energy - 0.1) 
        target_valence = max(0.3, target_valence - 0.1) 
        
    # --- Run Length (Structure/Danceability) ---
    run_length_bin = run_features.get("run_length_bin")
    target_danceability = 0.6
    if run_length_bin == "Long":
        # Focus on progressive rhythm for consistency
        target_danceability = 0.75 
        
    # --- Precipitation (Minor mood dampener) ---
    if run_features.get("precipitation", 0) > 0.5: # Over 0.5mm/h
        target_valence = max(0.2, target_valence - 0.15) # More moody/pensive
        
    
    # 3. Final Output (Convert to required Spotify format)
    return {
        "target_tempo": target_tempo,
        "target_energy": round(target_energy, 2),
        "target_valence": round(target_valence, 2),
        "target_danceability": round(target_danceability, 2),
        "limit": 30, # Get 30 tracks
        # Other optional seeds (for ML angle later)
        # "seed_genres": "workout,electronic", 
        # "min_popularity": 50 
    }


def generate_playlist_title(run_features: Dict[str, Any], audio_features: Dict[str, Any]) -> str:
    """Creates a descriptive playlist title."""
    temp_bin = run_features.get("temp_bin")
    avg_pace = avg_pace_min_per_km(run_features) # Use the helper for display
    
    # Determine music style keywords
    if audio_features["target_valence"] > 0.75:
        mood = "Upbeat"
    elif audio_features["target_valence"] < 0.4:
        mood = "Ambient"
    else:
        mood = "Driving"

    if audio_features["target_energy"] > 0.75:
        intensity = "Boost"
    else:
        intensity = "Tempo"
        
    # Format pace nicely (e.g., 5:15 min/km)
    minutes = int(avg_pace)
    seconds = int(round((avg_pace - minutes) * 60))
    pace_str = f"{minutes}:{seconds:02d}"
    
    return f"{mood} {intensity} Run | {temp_bin} | {pace_str} min/km"

TEMPO_MAP = {
    'slow': {'target_tempo': 120, 'target_energy': 0.4},  # Walking/Very Slow Jog
    'steady': {'target_tempo': 140, 'target_energy': 0.6}, # Moderate/Endurance Pace
    'fast': {'target_tempo': 160, 'target_energy': 0.8},  # Race Pace/Intervals
}

def generate_playlist_title_from_intent(run_intent: str) -> str:
    """Generates a playlist title based on the user's intent."""
    intent_map = {'fast': 'Speed', 'steady': 'Endurance', 'slow': 'Recovery'}
    run_type = intent_map.get(run_intent, 'Custom')
    return f"RunSound AI - {run_type} Run ({run_intent.capitalize()})"

def recommend_and_create_playlist():
    """Main function to run the recommendation pipeline based on user intent."""

    # --- 1. Determine Run Intent and Audio Targets ---
    if len(sys.argv) > 1:
        run_intent = sys.argv[1].lower()
    else:
        run_intent = 'steady'

    if run_intent in TEMPO_MAP:
        audio_targets = TEMPO_MAP[run_intent].copy()  # Use .copy() to avoid modifying the original
        print(f"Using user intent '{run_intent.capitalize()}' to set music targets.")
    else:
        print(f"Warning: Invalid intent '{run_intent}'. Falling back to 'Steady'.")
        run_intent = 'steady'
        audio_targets = TEMPO_MAP['steady'].copy()
    
    # Add limit to audio_targets
    audio_targets['limit'] = 30
        
    try:
        # 2. Initialize Spotify Client
        sp = SpotifyClient()
        
        # 3. Print Audio Targets
        print("\n--- Audio Feature Targets ---")
        print(json.dumps(audio_targets, indent=2))
        
        # 4. Build Seeds if not present
        print("\n--- Fetching Recommendations ---")
        if not any(k.startswith("seed_") for k in audio_targets.keys()):
            # Try genres first
            genres = sp.get_available_genre_seeds()
            
            preferred_by_intent = {
                "slow": ["acoustic", "ambient", "chill"],
                "steady": ["indie", "pop", "folk"],
                "fast": ["edm", "workout", "dance"]
            }.get(run_intent, ["pop", "indie"])

            chosen = [g for g in preferred_by_intent if g in genres][:2]
            
            if chosen:
                audio_targets["seed_genres"] = ",".join(chosen)
                print(f"Using seed_genres: {audio_targets['seed_genres']}")
            else:
                # Fallback to top artists - but let's try different time ranges
                print("No matching genre seeds found, trying user's top artists...")
                top_artists = None
                
                # Try different time ranges to find artists
                for time_range in ['medium_term', 'long_term', 'short_term']:
                    top_artists = sp.get_user_top_artists(limit=5, time_range=time_range)
                    if top_artists:
                        print(f"Found {len(top_artists)} artists from {time_range}")
                        break
                
                if top_artists:
                    # Use only up to 3 artists and make sure they're valid
                    seed_artists = top_artists[:3]
                    audio_targets["seed_artists"] = ",".join(seed_artists)
                    print(f"Using seed_artists: {audio_targets['seed_artists']}")
                else:
                    # Last resort: use any available genres
                    print("No top artists found, using fallback genres...")
                    if genres and len(genres) >= 2:
                        audio_targets["seed_genres"] = ",".join(genres[:2])
                        print(f"Fallback seed_genres: {audio_targets['seed_genres']}")
                    elif genres:
                        # If only one genre available, add a popular track as second seed
                        audio_targets["seed_genres"] = genres[0]
                        print(f"Using single genre seed: {audio_targets['seed_genres']}")
                    else:
                        # Ultimate fallback: hardcoded popular genres
                        audio_targets["seed_genres"] = "pop,rock"
                        print(f"Using hardcoded fallback genres: {audio_targets['seed_genres']}")

        # 5. Make the recommendation call (ONLY ONCE!)
        print(f"\n--- Calling Spotify API with params ---")
        print(json.dumps(audio_targets, indent=2))
        
        try:
            recommended_tracks = sp.get_recommendations(**audio_targets)
        except Exception as e:
            print(f"\nError getting recommendations: {e}")
            print(f"Audio targets used: {audio_targets}")
            
            # Try one more time with just genres as fallback
            print("\nAttempting fallback with simple genre seeds...")
            fallback_targets = {
                'target_tempo': audio_targets['target_tempo'],
                'target_energy': audio_targets['target_energy'],
                'seed_genres': 'pop,rock',
                'limit': 30
            }
            print(f"Fallback params: {fallback_targets}")
            recommended_tracks = sp.get_recommendations(**fallback_targets)
        
        if not recommended_tracks:
            print("Warning: Spotify returned no tracks. Try adjusting the rules or targets.")
            return

        track_uris = [t["uri"] for t in recommended_tracks]
        print(f"\n✓ Got {len(track_uris)} track recommendations")
        
        # 6. Generate Playlist Metadata
        playlist_title = generate_playlist_title_from_intent(run_intent)
        playlist_desc = (
            f"RunSound AI: Targets for a {run_intent.capitalize()} run. "
            f"Tempo: {audio_targets['target_tempo']} BPM, Energy: {audio_targets['target_energy']}"
        )
        
        # 7. Create Playlist and Add Tracks
        print(f"\n--- Creating Playlist: {playlist_title} ---")
        playlist_id = sp.create_playlist(
            name=playlist_title,
            description=playlist_desc,
            public=True
        )
        
        sp.add_tracks_to_playlist(playlist_id, track_uris)
        
        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
        
        # 8. Output and Cache
        output_metadata = {
            "run_intent": run_intent,
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "title": playlist_title,
            "track_count": len(track_uris),
            "audio_targets": audio_targets
        }
        
        OUT_PLAYLIST_METADATA_PATH.write_text(json.dumps(output_metadata, indent=2))
        
        print("\nSuccess! RunSound AI Playlist Generated.")
        print(f"Playlist URL: {playlist_url}")
        print(f"Metadata saved to: {OUT_PLAYLIST_METADATA_PATH}")

    except SpotifyAuthError as e:
        print(f"\nAuthentication Error: {e}")
        print("Ensure your SPOTIFY_REFRESH_TOKEN is valid in .env.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the recommender pipeline: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    recommend_and_create_playlist()