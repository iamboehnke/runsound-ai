"""
app.py
The CLI Orchestration script for RunSound AI.
Runs the entire pipeline (fetch, engineer, recommend) or the ML Recommender
for a planned run.
"""
import json
import webbrowser
import sys
from pathlib import Path
from typing import Dict, Any
import subprocess
from fetch_strava import get_latest_runs
from fetch_weather import fetch_weather
from feature_engineer import feature_engineer_runs, avg_pace_min_per_km
from ml_recommender import recommend_and_create_playlist

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
METADATA_PATH = DATA_DIR / "latest_playlist.json"


# --- Orchestration Function ---
def run_historical_data_pipeline() -> bool:
    """Executes the data fetching and engineering pipeline for historical data."""
    
    print("\n--- Starting RunSound AI Historical Data Pipeline ---")
    
    scripts = [
        ("Fetching Strava Data...", "src/fetch_strava.py", []),
        ("Fetching Weather Data...", "src/fetch_weather.py", []),
        ("Engineering Features...", "src/feature_engineer.py", []),
    ]

    for message, script_path, args in scripts:
        print(f"| RUNNING: {message}")
        try:
            # Construct the command: [sys.executable, script_path, arg1, arg2, ...]
            command = [sys.executable, "-X", "utf8", script_path] + args
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                check=True
            )
            # Only print a concise success message from the script
            print(f"| SUCCESS: {message}")
        except subprocess.CalledProcessError as e:
            print(f"\n| ERROR: Pipeline failed in {script_path}")
            print(f"| STDERR:\n{e.stderr.strip()}")
            return False
        except FileNotFoundError:
            print(f"\n| ERROR: Could not find required script: {script_path}.")
            return False
            
    print("--- Historical Data Pipeline Complete ---")
    return True


# --- Main Application Logic ---
def main():
    run_historical_data_pipeline()
    print("\n\n--- Starting ML-Powered Playlist Generation ---")
    recommend_and_create_playlist()
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            
            url = metadata.get("playlist_url", "URL not found")
            title = metadata.get("title", "RunSound AI Playlist")
            
            print("\n=============================================")
            print("FINAL STATUS: Playlist Ready (via ML Recommender)")
            print(f"   Playlist Title: {title}")
            print(f"   Playlist URL:   {url}")
            print("=============================================")

        except Exception as e:
            print(f"ERROR: Could not load final playlist metadata: {e}")
    else:
        print("\nERROR: ML Recommender finished, but final metadata file was not created. Check ml_recommender.py output.")
    input("\nRunSound AI pipeline finished. Press Enter to close.")


if __name__ == "__main__":
    main()