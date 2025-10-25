"""
spotify_client.py
Handles Spotify API authentication and basic utilities for RunSound AI.

Uses the Authorization Code Flow with Refresh Token for persistent access.

Requirements:
  - .env must contain:
      SPOTIFY_CLIENT_ID
      SPOTIFY_CLIENT_SECRET
      SPOTIFY_REFRESH_TOKEN (obtained after initial user authorization)
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIFY_REFRESH_TOKEN = os.getenv("SPOTIPY_REFRESH_TOKEN")
# SPOTIFY_REDIRECT_URI is only needed for the initial token exchange, not the refresh.

# Configuration
SPOTIFY_API_BASE = "https://api.spotify.com/v1" 
TOKEN_URL = "https://accounts.spotify.com/api/token" 
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

class SpotifyAuthError(RuntimeError):
    pass

class SpotifyClient:
    """A persistent client for the Spotify Web API using Refresh Tokens."""

    def __init__(self):
        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and SPOTIFY_REFRESH_TOKEN):
            raise SpotifyAuthError("Missing Spotify credentials (ID/SECRET/REFRESH_TOKEN) in .env.")
        self.access_token = self._refresh_access_token()
        self.user_id = self.get_user_id()
        print("SpotifyClient initialized and token refreshed.")

    def _refresh_access_token(self) -> str:
        """Use refresh token to get a new access token."""
        auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        headers = {
            "Authorization": "Basic " + base64.b64encode(auth_str.encode()).decode(),
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": SPOTIFY_REFRESH_TOKEN,
        }
        
        print(f"Refreshing token with client_id: {SPOTIFY_CLIENT_ID[:10]}...")  # Only show first 10 chars
        
        r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=15)
        
        if r.status_code != 200:
            raise SpotifyAuthError(f"Failed to refresh token: {r.status_code} {r.text}. Check your SPOTIFY_REFRESH_TOKEN.")
        
        token_info = r.json()
        access_token = token_info.get("access_token")
        
        print(f"✓ Token refreshed successfully. Token starts with: {access_token[:20]}...")
        print(f"✓ Token scopes: {token_info.get('scope', 'NOT PROVIDED')}")
        
        return access_token

    def _headers(self):
        """Standard headers for API calls."""
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def get_user_id(self) -> str:
        """Fetch the current authenticated user's ID."""
        url = f"{SPOTIFY_API_BASE}/me"
        r = requests.get(url, headers=self._headers(), timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to get user info: {r.status_code} {r.text}")
        return r.json()["id"]

    # --- Core Music Recommendation Methods (Used by recommender.py) ---
    
    def get_available_genre_seeds(self) -> List[str]:
        """Return Spotify's allowed seed genres (useful to pick valid seed_genres)."""
        url = f"{SPOTIFY_API_BASE}/recommendations/available-genre-seeds"
        r = requests.get(url, headers=self._headers(), timeout=15)
        if r.status_code != 200:
            print(f"Warning: Failed to fetch genre seeds: {r.status_code} {r.text}")
            return []
        return r.json().get("genres", [])

    def get_user_top_artists(self, limit: int = 5, time_range: str = "short_term") -> List[str]:
        """
        Return top artist IDs for the current user (useful as seed_artists).
        time_range can be 'short_term', 'medium_term', or 'long_term'.
        """
        url = f"{SPOTIFY_API_BASE}/me/top/artists"
        params = {"limit": limit, "time_range": time_range}
        r = requests.get(url, headers=self._headers(), params=params, timeout=15)
        if r.status_code != 200:
            print(f"Warning: Failed to fetch user's top artists: {r.status_code} {r.text}")
            return []
        return [a["id"] for a in r.json().get("items", [])]

    def get_recommendations(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Uses the Spotify Get Recommendations endpoint.
        Requires at least one seed parameter.
        """
        has_seed = any(k.startswith("seed_") for k in kwargs.keys())
        if not has_seed:
            raise ValueError(
                "Spotify recommendations require at least one seed parameter: "
                "seed_artists, seed_tracks, or seed_genres."
            )

        url = f"{SPOTIFY_API_BASE}/recommendations"
        r = requests.get(url, headers=self._headers(), params=kwargs, timeout=15)

        if r.status_code != 200:
            # Print the actual request params for debugging
            print(f"Request params: {kwargs}")
            print(f"Response: {r.text}")
            raise RuntimeError(f"Spotify recommendations failed: {r.status_code} - {r.text}")

        return r.json().get("tracks", [])

    def get_audio_features_for_tracks(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches detailed audio features (tempo, energy, valence, etc.) for a list of track IDs."""
        if not track_ids: return []
        track_id_str = ",".join(track_ids[:100]) # Max 100 tracks
        
        url = f"{SPOTIFY_API_BASE}/audio-features"
        params = {"ids": track_id_str}
        
        r = requests.get(url, headers=self._headers(), params=params, timeout=15)
        if r.status_code != 200:
            print(f"Warning: Failed to fetch audio features: {r.status_code} {r.text}")
            return []
        
        return r.json().get("audio_features", [])
    
    def test_recommendations_endpoint(self):
        """Test if the recommendations endpoint is accessible."""
        url = f"{SPOTIFY_API_BASE}/recommendations/available-genre-seeds"
        print(f"\nTesting endpoint: {url}")
        print(f"Using token: {self.access_token[:20]}...")
        
        r = requests.get(url, headers=self._headers(), timeout=15)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:200]}")  # First 200 chars
        
        if r.status_code == 200:
            print("Recommendations endpoint is accessible!")
            return True
        else:
            print("Recommendations endpoint returned error")
            return False

    # --- Playlist Creation Methods ---

    def create_playlist(self, name: str, description: str = "", public: bool = True) -> str:
        """Create a new playlist and return its ID."""
        url = f"{SPOTIFY_API_BASE}/users/{self.user_id}/playlists"
        data = {"name": name, "description": description, "public": public}
        r = requests.post(url, headers=self._headers(), json=data, timeout=15)
        if r.status_code != 201: # 201 Created
            raise RuntimeError(f"Failed to create playlist: {r.status_code} {r.text}")
        return r.json()["id"]

    def add_tracks_to_playlist(self, playlist_id: str, track_uris: List[str]):
        """Adds a list of track URIs to the specified playlist ID."""
        url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks"
        
        # Spotify limit is 100 tracks per request
        for i in range(0, len(track_uris), 100):
            batch = track_uris[i:i + 100]
            r = requests.post(url, headers=self._headers(), json={"uris": batch}, timeout=15)
            if r.status_code not in (201, 200):
                raise RuntimeError(f"Failed to add tracks: {r.status_code} {r.text}")
        
        print(f"Added {len(track_uris)} tracks to playlist {playlist_id}.")

if __name__ == "__main__":
    print("Running Spotify client demo (requires SPOTIFY_REFRESH_TOKEN in .env)...")
    try:
        sp = SpotifyClient()
        print(f"Authenticated as User ID: {sp.user_id}")
        
        # Test the recommendations endpoint
        sp.test_recommendations_endpoint()
        
        # Example: Get recommendations for a specific tempo/energy
        print("\n--- Testing Recommendations Endpoint ---")
        recommended_tracks = sp.get_recommendations(
            target_tempo=150,
            target_energy=0.8,
            seed_genres="pop,rock",
            limit=5
        )
        if recommended_tracks:
            print(f"Found {len(recommended_tracks)} recommended tracks.")
            print(f"Example track: {recommended_tracks[0].get('name')} by {recommended_tracks[0].get('artists')[0].get('name')}")
        
    except SpotifyAuthError as e:
        print(f"\nAuthentication Error: {e}")
        print("To fix this, you must complete the initial Spotify Authorization Code Flow to get a SPOTIFY_REFRESH_TOKEN.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()