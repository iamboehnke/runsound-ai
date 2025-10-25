"""
get_new_spotify_token.py
One-time script to get a Spotify refresh token with all required scopes.
Run this once, then copy the refresh token to your .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Check if spotipy is installed
try:
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("ERROR: spotipy is not installed.")
    print("Install it with: pip install spotipy")
    exit(1)

# All the scopes we need for RunSound AI
REQUIRED_SCOPES = [
    "user-read-private",
    "user-read-email",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-read-playback-state",  # This is needed for audio features
]

scope_string = " ".join(REQUIRED_SCOPES)

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

if not client_id or not client_secret:
    print("ERROR: SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be in your .env file")
    exit(1)

print("="*60)
print("Spotify Token Generator for RunSound AI")
print("="*60)
print(f"\nClient ID: {client_id[:10]}...")
print(f"Required scopes: {scope_string}")
print("\nThis will open your browser for authorization...")
print("After authorizing, you'll be redirected to localhost.")
print("Copy the ENTIRE URL from your browser and paste it back here.")
print("="*60)

sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri="http://127.0.0.1:8888/callback",
    scope=scope_string,
    open_browser=True
)

# Get the authorization URL
auth_url = sp_oauth.get_authorize_url()
print(f"\nIf browser doesn't open, visit: {auth_url}")

# This will open the browser and wait for the callback
token_info = sp_oauth.get_access_token(as_dict=True)

if token_info:
    refresh_token = token_info['refresh_token']
    
    print("\n" + "="*60)
    print("SUCCESS! Token Generated")
    print("="*60)
    print("\nAdd this line to your .env file:")
    print(f"\nSPOTIFY_REFRESH_TOKEN={refresh_token}")
    print("\n(Replace the old SPOTIPY_REFRESH_TOKEN line)")
    print("="*60)
    print(f"\nToken scopes: {token_info.get('scope', 'N/A')}")
    print("="*60)
else:
    print("\nFailed to get token. Please try again.")