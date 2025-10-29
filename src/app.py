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

# NOTE: The ML Recommender is a self-contained script that asks the user for
# run parameters, so we can replace the user prompt and the old recommender call.

# Import pipeline components - Keeping feature_engineer for historical run data,
# but we will bypass its use for the *current* run's music prediction.
from fetch_strava import get_latest_runs
from fetch_weather import fetch_weather
from feature_engineer import feature_engineer_runs, avg_pace_min_per_km
# --- START CHANGE 1: Update Import ---
from ml_recommender import recommend_and_create_playlist_ml # Import the new ML function
# --- END CHANGE 1 ---

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
METADATA_PATH = DATA_DIR / "latest_playlist.json"


# --- Orchestration Function ---

# NOTE: The old 'get_user_run_intent' is removed as the ML recommender prompts
# the user for all necessary details (pace, distance, run type, etc.) itself.

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
    # It is good practice to run the historical data pipeline first to ensure
    # the necessary files (like ml_featured_runs.json) are up-to-date for the ML model.
    run_historical_data_pipeline()

    # --- START CHANGE 2 & 3: Replace old call with ML Recommender ---
    # The ML recommender function is now the central entry point for playlist generation.
    print("\n\n--- Starting ML-Powered Playlist Generation ---")
    
    # This function handles user input, model loading, prediction, track filtering,
    # playlist creation, metadata saving, and browser opening internally.
    recommend_and_create_playlist_ml()

    # The rest of the main function is now redundant as the ml_recommender handles
    # success/URL display, but we keep the final input for console stability.
    
    # We can reload the metadata saved by the ml_recommender just to confirm the end result.
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

    # Keep the console window open after completion for the user to see success/details
    input("\nRunSound AI pipeline finished. Press Enter to close.")


if __name__ == "__main__":
    main()