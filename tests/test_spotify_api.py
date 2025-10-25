"""
test_spotify_api.py
Direct test of Spotify API endpoints
"""
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

import requests
from dotenv import load_dotenv
from spotify_client import SpotifyClient

load_dotenv()

def test_endpoints():
    sp = SpotifyClient()
    
    base_url = "https://api.spotify.com/v1"
    headers = {"Authorization": f"Bearer {sp.access_token}"}
    
    # Test different endpoints
    endpoints = [
        "/me",
        "/me/top/artists?limit=1",
        "/recommendations/available-genre-seeds",
        "/search?q=rock&type=track&limit=1",
    ]
    
    print("\n" + "="*60)
    print("TESTING SPOTIFY API ENDPOINTS")
    print("="*60)
    
    for endpoint in endpoints:
        url = base_url + endpoint
        print(f"\nTesting: {endpoint}")
        print(f"Full URL: {url}")
        
        r = requests.get(url, headers=headers, timeout=15)
        
        print(f"Status: {r.status_code}")
        print(f"Response length: {len(r.text)} chars")
        
        if r.status_code == 200:
            print(f"✓ SUCCESS")
            # Print first 150 chars of response
            print(f"Response preview: {r.text[:150]}...")
        else:
            print(f"✗ FAILED")
            print(f"Response: {r.text}")
            print(f"Headers: {dict(r.headers)}")
    
    # Now try the recommendations endpoint with full URL
    print("\n" + "="*60)
    print("TESTING RECOMMENDATIONS DIRECTLY")
    print("="*60)
    
    rec_url = "https://api.spotify.com/v1/recommendations"
    params = {
        "seed_genres": "pop,rock",
        "limit": 5
    }
    
    print(f"\nURL: {rec_url}")
    print(f"Params: {params}")
    
    r = requests.get(rec_url, headers=headers, params=params, timeout=15)
    
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
    print(f"Response headers: {dict(r.headers)}")

if __name__ == "__main__":
    test_endpoints()