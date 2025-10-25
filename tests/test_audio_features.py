"""
test_audio_features.py
Test if we can access audio features with the new token
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

import requests
from spotify_client import SpotifyClient

def test_audio_features():
    sp = SpotifyClient()
    
    print("\n" + "="*60)
    print("TESTING AUDIO FEATURES ACCESS")
    print("="*60)
    
    # First, search for a track to get a valid track ID
    url = "https://api.spotify.com/v1/search"
    params = {'q': 'running', 'type': 'track', 'limit': 5}
    
    r = requests.get(url, headers=sp._headers(), params=params, timeout=15)
    
    if r.status_code != 200:
        print(f"❌ Search failed: {r.status_code}")
        return
    
    tracks = r.json()['tracks']['items']
    track_ids = [t['id'] for t in tracks]
    
    print(f"\n✓ Found {len(track_ids)} tracks")
    print(f"Track IDs: {track_ids[:3]}...")
    
    # Test 1: Single track audio features
    print("\n--- Test 1: Single Track Audio Features ---")
    single_track_id = track_ids[0]
    url_single = f"https://api.spotify.com/v1/audio-features/{single_track_id}"
    
    print(f"URL: {url_single}")
    r = requests.get(url_single, headers=sp._headers(), timeout=15)
    
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:300]}...")
    
    if r.status_code == 200:
        print("✓ Single track audio features WORKS!")
    
    # Test 2: Multiple tracks audio features
    print("\n--- Test 2: Multiple Tracks Audio Features ---")
    url_multi = "https://api.spotify.com/v1/audio-features"
    params_multi = {"ids": ",".join(track_ids)}
    
    print(f"URL: {url_multi}")
    print(f"Params: {params_multi}")
    r = requests.get(url_multi, headers=sp._headers(), params=params_multi, timeout=15)
    
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}...")
    
    if r.status_code == 200:
        print("✓ Multiple tracks audio features WORKS!")
        features = r.json().get('audio_features', [])
        if features and features[0]:
            print(f"\nSample audio features:")
            print(f"  Tempo: {features[0].get('tempo')}")
            print(f"  Energy: {features[0].get('energy')}")
            print(f"  Valence: {features[0].get('valence')}")
    
    # Test 3: Try with market parameter
    print("\n--- Test 3: With Market Parameter ---")
    params_market = {"ids": ",".join(track_ids[:3]), "market": "DK"}
    
    r = requests.get(url_multi, headers=sp._headers(), params=params_market, timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:300]}...")

if __name__ == "__main__":
    test_audio_features()