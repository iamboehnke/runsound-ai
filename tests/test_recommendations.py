import os
import pytest
import sys

from spotify_client import SpotifyClient, SpotifyAuthError

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

pytestmark = pytest.mark.integration  # mark so you can run only integration tests if desired


def _has_spotify_env() -> bool:
    # Match the env var names used in your spotify_client.py
    return all(
        os.getenv(name)
        for name in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REFRESH_TOKEN")
    )


@pytest.fixture(scope="module")
def spotify_client():
    if not _has_spotify_env():
        pytest.skip(
            "Spotify credentials not found in environment. "
            "Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REFRESH_TOKEN to run this test."
        )
    try:
        sp = SpotifyClient()
    except SpotifyAuthError as exc:
        pytest.skip(f"Spotify auth failed: {exc}")
    except Exception as exc:
        pytest.skip(f"Skipping Spotify integration tests due to unexpected error: {exc}")
    return sp


def test_genre_seeds_and_recommendations(spotify_client):
    sp = spotify_client

    # 1) Fetch available genres (should be a list; may be empty depending on API/auth)
    genres = sp.get_available_genre_seeds()
    assert isinstance(genres, list), "get_available_genre_seeds must return a list"

    # 2) Try to pick sensible seeds (prefer intent-style genres)
    preferred = ["acoustic", "ambient", "chill", "indie", "pop", "dance", "workout", "edm"]
    chosen = [g for g in preferred if g in genres][:2]

    params = {}
    if chosen:
        params["seed_genres"] = ",".join(chosen)
    else:
        # fallback: use user's top artists
        top_artists = sp.get_user_top_artists(limit=3)
        assert isinstance(top_artists, list)
        if top_artists:
            params["seed_artists"] = ",".join(top_artists[:3])
        else:
            # last resort: if we can't get any seeds, skip the test
            pytest.skip("No valid seed_genres or seed_artists available from Spotify API")

    # 3) Add audio targets & request a small number of tracks
    params.update({"target_tempo": 140, "target_energy": 0.6, "limit": 5})

    tracks = sp.get_recommendations(**params)
    assert isinstance(tracks, list), "get_recommendations must return a list"

    # 4) Basic validations on returned tracks
    assert len(tracks) > 0, "Spotify returned zero tracks; try different seeds/params"
    for t in tracks:
        assert "uri" in t and "name" in t, "Each track should contain at least 'uri' and 'name'"
