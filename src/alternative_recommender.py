"""
Alternative recommendation engine that works without the /recommendations endpoint.
Uses search + audio features filtering instead.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import random

from spotify_client import SpotifyClient, SpotifyAuthError

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PLAYLIST_METADATA_PATH = DATA_DIR / "latest_playlist.json"

TEMPO_MAP = {
    'slow': {'target_tempo': 120, 'target_energy': 0.4, 'target_valence': 0.5},
    'steady': {'target_tempo': 140, 'target_energy': 0.6, 'target_valence': 0.6},
    'fast': {'target_tempo': 160, 'target_energy': 0.8, 'target_valence': 0.7},
}

# Search queries for different intents
SEARCH_QUERIES = {
    'slow': ['chill running', 'easy jog', 'recovery run', 'acoustic workout', 'slow tempo'],
    'steady': ['running playlist', 'jogging music', 'endurance run', 'workout mix', 'running songs'],
    'fast': ['high energy workout', 'fast running', 'interval training', 'tempo run', 'speed workout']
}


def generate_playlist_title_from_intent(run_intent: str) -> str:
    """Generates a playlist title based on the user's intent."""
    intent_map = {'fast': 'Speed', 'steady': 'Endurance', 'slow': 'Recovery'}
    run_type = intent_map.get(run_intent, 'Custom')
    return f"RunSound AI - {run_type} Run ({run_intent.capitalize()})"


def search_tracks_by_query(sp: SpotifyClient, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search for tracks using a query string."""
    url = f"https://api.spotify.com/v1/search"
    params = {
        'q': query,
        'type': 'track',
        'limit': limit
    }
    
    import requests
    r = requests.get(url, headers=sp._headers(), params=params, timeout=15)
    
    if r.status_code != 200:
        print(f"Warning: Search failed for query '{query}': {r.status_code}")
        return []
    
    data = r.json()
    return data.get('tracks', {}).get('items', [])


def filter_tracks_by_audio_features(
    sp: SpotifyClient, 
    tracks: List[Dict[str, Any]], 
    target_tempo: int,
    target_energy: float,
    target_valence: float,
    tempo_tolerance: int = 20,
    energy_tolerance: float = 0.3,
    valence_tolerance: float = 0.3
) -> List[Dict[str, Any]]:
    """Filter tracks based on audio features."""
    if not tracks:
        return []
    
    # Get track IDs
    track_ids = [t['id'] for t in tracks if t.get('id')]
    
    # Fetch audio features
    audio_features = sp.get_audio_features_for_tracks(track_ids)
    
    # Create a mapping of track_id -> features
    features_map = {f['id']: f for f in audio_features if f}
    
    # Filter tracks
    filtered = []
    for track in tracks:
        track_id = track.get('id')
        if not track_id or track_id not in features_map:
            continue
        
        features = features_map[track_id]
        tempo = features.get('tempo', 0)
        energy = features.get('energy', 0)
        valence = features.get('valence', 0)
        
        # Check if features match our targets (within tolerance)
        tempo_match = abs(tempo - target_tempo) <= tempo_tolerance
        energy_match = abs(energy - target_energy) <= energy_tolerance
        valence_match = abs(valence - target_valence) <= valence_tolerance
        
        if tempo_match and energy_match and valence_match:
            # Add audio features to track object for debugging
            track['_audio_features'] = {
                'tempo': tempo,
                'energy': energy,
                'valence': valence
            }
            filtered.append(track)
    
    return filtered


def recommend_and_create_playlist():
    """Main function to run the alternative recommendation pipeline."""
    
    # --- 1. Determine Run Intent and Audio Targets ---
    if len(sys.argv) > 1:
        run_intent = sys.argv[1].lower()
    else:
        run_intent = 'steady'
    
    if run_intent not in TEMPO_MAP:
        print(f"Warning: Invalid intent '{run_intent}'. Falling back to 'Steady'.")
        run_intent = 'steady'
    
    audio_targets = TEMPO_MAP[run_intent]
    
    print(f"Using user intent '{run_intent.capitalize()}' to set music targets.")
    print("\n--- Audio Feature Targets ---")
    print(json.dumps(audio_targets, indent=2))
    
    try:
        # 2. Initialize Spotify Client
        sp = SpotifyClient()
        
        # 3. Search for tracks using multiple queries
        print("\n--- Searching for Tracks ---")
        all_tracks = []
        queries = SEARCH_QUERIES.get(run_intent, ['running music'])
        
        for query in queries:
            print(f"Searching: '{query}'...")
            tracks = search_tracks_by_query(sp, query, limit=50)
            all_tracks.extend(tracks)
            print(f"  Found {len(tracks)} tracks")
        
        print(f"\nTotal tracks found: {len(all_tracks)}")
        
        # Remove duplicates
        unique_tracks = {t['id']: t for t in all_tracks if t.get('id')}
        all_tracks = list(unique_tracks.values())
        print(f"Unique tracks: {len(all_tracks)}")
        
        # 4. Filter by audio features
        print("\n--- Filtering by Audio Features ---")
        filtered_tracks = filter_tracks_by_audio_features(
            sp,
            all_tracks,
            target_tempo=audio_targets['target_tempo'],
            target_energy=audio_targets['target_energy'],
            target_valence=audio_targets['target_valence'],
            tempo_tolerance=25,  # +/- 25 BPM
            energy_tolerance=0.25,  # +/- 0.25
            valence_tolerance=0.35  # +/- 0.35 (more lenient)
        )
        
        print(f"Tracks matching criteria: {len(filtered_tracks)}")
        
        if len(filtered_tracks) < 15:
            print("Warning: Not enough matching tracks. Relaxing filters...")
            # Relax filters
            filtered_tracks = filter_tracks_by_audio_features(
                sp,
                all_tracks,
                target_tempo=audio_targets['target_tempo'],
                target_energy=audio_targets['target_energy'],
                target_valence=audio_targets['target_valence'],
                tempo_tolerance=40,
                energy_tolerance=0.4,
                valence_tolerance=0.5
            )
            print(f"Tracks with relaxed filters: {len(filtered_tracks)}")
        
        if not filtered_tracks:
            print("Error: No tracks found matching criteria. Try a different intent.")
            return
        
        # 5. Select up to 30 tracks randomly
        selected_tracks = random.sample(filtered_tracks, min(30, len(filtered_tracks)))
        track_uris = [t['uri'] for t in selected_tracks]
        
        # Print some examples
        print("\n--- Sample Tracks ---")
        for i, track in enumerate(selected_tracks[:5], 1):
            artist = track['artists'][0]['name'] if track.get('artists') else 'Unknown'
            features = track.get('_audio_features', {})
            print(f"{i}. {track['name']} - {artist}")
            print(f"   Tempo: {features.get('tempo', 'N/A')}, Energy: {features.get('energy', 'N/A')}, Valence: {features.get('valence', 'N/A')}")
        
        # 6. Create Playlist
        playlist_title = generate_playlist_title_from_intent(run_intent)
        playlist_desc = (
            f"RunSound AI: Curated for a {run_intent.capitalize()} run. "
            f"Target Tempo: {audio_targets['target_tempo']} BPM, Energy: {audio_targets['target_energy']}"
        )
        
        print(f"\n--- Creating Playlist: {playlist_title} ---")
        playlist_id = sp.create_playlist(
            name=playlist_title,
            description=playlist_desc,
            public=True
        )
        
        sp.add_tracks_to_playlist(playlist_id, track_uris)
        
        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
        
        # 7. Save metadata
        output_metadata = {
            "run_intent": run_intent,
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "title": playlist_title,
            "track_count": len(track_uris),
            "audio_targets": audio_targets
        }
        
        OUT_PLAYLIST_METADATA_PATH.write_text(json.dumps(output_metadata, indent=2))
        
        print("\n✅ Success! RunSound AI Playlist Generated.")
        print(f"Playlist URL: {playlist_url}")
        print(f"Metadata saved to: {OUT_PLAYLIST_METADATA_PATH}")
        
    except SpotifyAuthError as e:
        print(f"\nAuthentication Error: {e}")
        print("Ensure your SPOTIFY_REFRESH_TOKEN is valid in .env.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

def recommend_and_create_playlist_simple():
    """Simplified version that doesn't require audio features access."""
    
    # --- 1. Determine Run Intent ---
    if len(sys.argv) > 1:
        run_intent = sys.argv[1].lower()
    else:
        run_intent = 'steady'
    
    if run_intent not in TEMPO_MAP:
        print(f"Warning: Invalid intent '{run_intent}'. Falling back to 'Steady'.")
        run_intent = 'steady'
    
    audio_targets = TEMPO_MAP[run_intent]
    
    print(f"Using user intent '{run_intent.capitalize()}' to set music targets.")
    print("\n--- Audio Feature Targets ---")
    print(json.dumps(audio_targets, indent=2))
    
    try:
        # 2. Initialize Spotify Client
        sp = SpotifyClient()
        
        # 3. Search for tracks
        print("\n--- Searching for Tracks ---")
        all_tracks = []
        queries = SEARCH_QUERIES.get(run_intent, ['running music'])
        
        for query in queries:
            print(f"Searching: '{query}'...")
            tracks = search_tracks_by_query(sp, query, limit=50)
            all_tracks.extend(tracks)
            print(f"  Found {len(tracks)} tracks")
        
        # Remove duplicates
        unique_tracks = {t['id']: t for t in all_tracks if t.get('id')}
        all_tracks = list(unique_tracks.values())
        
        print(f"\nTotal unique tracks: {len(all_tracks)}")
        
        # 4. Select 30 random tracks (no filtering)
        import random
        selected_tracks = random.sample(all_tracks, min(30, len(all_tracks)))
        track_uris = [t['uri'] for t in selected_tracks]
        
        # 5. Create Playlist
        playlist_title = generate_playlist_title_from_intent(run_intent)
        playlist_desc = (
            f"RunSound AI: Curated for a {run_intent.capitalize()} run. "
            f"Target Tempo: {audio_targets['target_tempo']} BPM"
        )
        
        print(f"\n--- Creating Playlist: {playlist_title} ---")
        playlist_id = sp.create_playlist(
            name=playlist_title,
            description=playlist_desc,
            public=True
        )
        
        sp.add_tracks_to_playlist(playlist_id, track_uris)
        
        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
        
        # 6. Save metadata
        output_metadata = {
            "run_intent": run_intent,
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "title": playlist_title,
            "track_count": len(track_uris),
            "audio_targets": audio_targets
        }
        
        OUT_PLAYLIST_METADATA_PATH.write_text(json.dumps(output_metadata, indent=2))
        
        print("\n✅ Success! RunSound AI Playlist Generated.")
        print(f"Playlist URL: {playlist_url}")
        print(f"Tracks: {len(track_uris)}")
        print(f"Metadata saved to: {OUT_PLAYLIST_METADATA_PATH}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        sp = SpotifyClient()
        test_tracks = search_tracks_by_query(sp, "running", limit=5)
        if test_tracks:
            test_ids = [t['id'] for t in test_tracks[:3]]
            features = sp.get_audio_features_for_tracks(test_ids)
            if features:
                print("Audio features available - using full filtering")
                recommend_and_create_playlist()
            else:
                print("Audio features not available - using simple search")
                recommend_and_create_playlist_simple()
        else:
            recommend_and_create_playlist_simple()
    except Exception as e:
        print(f"Falling back to simple mode: {e}")
        recommend_and_create_playlist_simple()