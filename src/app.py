"""
app.py
The CLI Orchestration script for RunSound AI.
Runs the entire pipeline (fetch, engineer, recommend) and opens the result.
"""

import json
import webbrowser
import sys
from pathlib import Path
from typing import Dict, Any
import subprocess

# We now import the pipeline components directly to avoid 'subprocess' issues
# If you get errors about modules not being found, you may need to adjust the path:
# sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Import pipeline functions
from fetch_strava import get_latest_runs
from fetch_weather import fetch_weather
from feature_engineer import feature_engineer_runs, avg_pace_min_per_km
from recommender import recommend_and_create_playlist # We will simplify this function next

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
METADATA_PATH = DATA_DIR / "latest_playlist.json"

# --- Orchestration Function ---
def get_user_run_intent() -> str:
    """Prompts the user for their desired run tempo or pace."""
    print("\n--- NEW RUN SETUP ---")
    
    # We will ask for a high-level intent (Fast, Steady, Slow) for simplicity
    # but you could easily extend this to accept a specific pace like '5:30'
    while True:
        intent = input("What kind of run are you planning? (Type 'Fast', 'Steady', or 'Slow'): ").strip().lower()
        if intent in ['fast', 'steady', 'slow']:
            return intent
        print("Invalid input. Please type 'Fast', 'Steady', or 'Slow'.")

def run_pipeline(run_intent: str) -> bool:
    """Executes the entire RunSound AI data pipeline sequentially."""
    
    print("\n--- Starting RunSound AI Pipeline ---")
    print(f"Goal: Generate playlist for a '{run_intent.capitalize()}' run.")
    
    # 1. Fetch Strava Data (Needed to get run history for the recommender/engineer scripts)
    # 2. Fetch Weather Data (Still useful for history)
    # 3. Engineer Features (Still useful for history, but music generation will be overridden)
    # 4. Generate Playlist (NEW: Pass the run_intent)

    scripts = [
        ("Fetching Strava Data...", "src/fetch_strava.py", []),
        ("Fetching Weather Data...", "src/fetch_weather.py", []),
        ("Engineering Features...", "src/feature_engineer.py", []),
        # NEW: Pass the run intent as a command-line argument to recommender.py
        ("Generating Spotify Playlist...", "src/alternative_recommender.py", [run_intent]),
    ]

    for message, script_path, args in scripts:
        print(f"| RUNNING: {message}")
        try:
            # Construct the command: [sys.executable, script_path, arg1, arg2, ...]
            # Use the previous encoding fix here for reliability
            command = [sys.executable, "-X", "utf8", script_path] + args
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', # Explicitly decode output
                check=True
            )
            print(f"| SUCCESS: {message} Output:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"\n| ERROR: Pipeline failed in {script_path}")
            print(f"| STDOUT:\n{e.stdout}")
            print(f"| STDERR:\n{e.stderr}")
            input("\nPress Enter to exit.") # Pause to see the error
            return False
        except FileNotFoundError:
            print(f"\n| ERROR: Could not find required script: {script_path}.")
            input("\nPress Enter to exit.")
            return False
            
    return True

# --- Main Application Logic ---

def main():
    user_intent = get_user_run_intent()
    if run_pipeline(user_intent):
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                
                url = metadata.get("playlist_url")
                title = metadata.get("title", "RunSound AI Playlist")
                
                print("\n=============================================")
                print("RunSound AI Playlist Generated Successfully!")
                print(f"   Playlist Title: {title}")
                print(f"   Playlist URL:   {url}")
                print("=============================================")
                
                # --- The key replacement for the Streamlit UI ---
                print("\nOpening playlist in your web browser...")
                webbrowser.open(url)
                
            except Exception as e:
                print(f"ERROR: Could not load final playlist metadata: {e}")
        else:
            print("\nERROR: Pipeline finished, but final metadata file was not created.")

    # Keep the console window open after completion for the user to see success/details
    input("\nRunSound AI pipeline finished. Press Enter to close.")


if __name__ == "__main__":
    main()