# RunSound AI

**Machine Learning-Powered Music Recommendations for Runners**

A system that analyzes your Strava training history and generates personalized Spotify playlists based on your planned workout. Uses machine learning to predict optimal music characteristics (tempo, energy, valence) from contextual features like pace, distance, weather, and training load.

---

<img width="990" height="716" alt="Screenshot 2" src="https://github.com/user-attachments/assets/1bb082da-bb2d-4c6a-acd0-664627bdb00d" />
<img width="990" height="762" alt="Screenshot 1" src="https://github.com/user-attachments/assets/85fb2ddd-28d3-43c1-8601-79243f2eb23f" />
<img width="1260" height="932" alt="Screenshot 4" src="https://github.com/user-attachments/assets/b33e6326-80f1-491d-937e-2cdd405a3998" />
<img width="966" height="623" alt="Screenshot 3" src="https://github.com/user-attachments/assets/2d798a41-f297-4655-ba63-f1f592fd41ef" />

---

## Overview

RunSound AI trains a Random Forest model on your historical running data to understand your patterns. When you plan a run, it predicts the ideal music profile and generates a 30-track Spotify playlist tailored to that specific workout.

The system considers:
- Run type (interval, tempo, easy, long, race, steady)
- Target pace and distance
- Current weather conditions
- Recent training load and fatigue
- Time of day
- Your historical preferences

**Key Capabilities:**

**Intelligent Pace Suggestions**
Analyzes your recent runs of the same type and suggests pace ranges based on actual performance data.

**Fatigue Detection**
Tracks weekly mileage and warns when training load is high. Adjusts music recommendations based on accumulated fatigue.

**Weather Integration**
Automatically fetches current weather forecast. No manual input required. Adapts music to conditions.

**Progressive Playlists**
For long runs (12km+), creates structured progression: warmup phase, steady main section, finish strong.

**Context Awareness**
Adjusts recommendations based on time of day, run type, weather conditions, and training history.

---

## Technical Architecture

### Data Pipeline

**Collection**
- Fetches Strava activities via API
- Matches each run with historical weather data (Open-Meteo API)
- Caches locally for offline model training

**Feature Engineering**
Extracts 15+ features per run:
- Core metrics: pace, distance, elevation gain
- Weather: temperature, precipitation, wind, humidity
- Temporal: time of day, day of week
- Historical: weekly mileage, pace consistency
- Classification: run type detection from activity name and metrics

**Model Training**
Random Forest Regressor predicting three targets:
- Target Tempo (BPM)
- Energy Level (0-1 scale)
- Valence/Positivity (0-1 scale)

Performance metrics (50+ training runs):
- Tempo: R² ~0.75, MAE ~10 BPM
- Energy: R² ~0.70, MAE ~0.08
- Valence: R² ~0.65, MAE ~0.10

**Prediction & Generation**
1. User inputs planned run details
2. System auto-fetches current weather
3. ML model predicts optimal music features
4. Generates intelligent Spotify search queries
5. Filters and structures tracks
6. Creates Spotify playlist

### Technology Stack

**Backend**
- Python 3.9+
- scikit-learn (machine learning)
- pandas (data processing)
- Flask (API/web interface)

**APIs**
- Strava API (running data)
- Spotify API (music search and playlists)
- Open-Meteo API (weather forecasts)

**Model**
- Random Forest Regression
- StandardScaler for feature normalization
- LabelEncoder for categorical features
- Pickle for model persistence

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Strava account with API access
- Spotify Premium account
- 500MB disk space

### Setup

**1. Clone Repository**
```bash
git clone https://github.com/yourusername/runsound-ai.git
cd runsound-ai
```

**2. Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure API Credentials**

Create `.env` file in project root:

```env
# Strava API
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_REFRESH_TOKEN=your_refresh_token

# Spotify API
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REFRESH_TOKEN=your_refresh_token
```

**Get Strava Credentials:**
1. Create application at https://www.strava.com/settings/api
2. Note Client ID and Client Secret
3. Complete OAuth flow to obtain refresh token

**Get Spotify Credentials:**
1. Create application at https://developer.spotify.com/dashboard
2. Add redirect URI: `http://127.0.0.1:8888/callback`
3. Run token generator:
```bash
python get_new_spotify_token.py
```

**5. Initial Data Collection**

```bash
# Fetch Strava running history
python src/fetch_strava.py

# Match runs with weather data
python src/fetch_weather.py

# Engineer ML features
python src/feature_engineer.py
```

Output files:
- `data/latest_runs.json` - Strava activities
- `data/run_weather.json` - Weather-matched runs  
- `data/ml_featured_runs.json` - ML-ready features

**6. Train Model**

```bash
python src/train_music_model.py
```

Creates: `data/models/music_recommender_model.pkl`

Minimum 10-15 runs required. Model performance improves significantly with 30+ runs.

---

## Usage

### Generate Playlist (CLI)

```bash
python src/ml_recommender.py
```

Interactive session:
```
Run type: tempo
Target pace (e.g., '5:30'): 5:15
Distance (km): 10
Expected temperature (°C, default 15): 
Time (default Morning): 

Analyzing your recent training...
  Recent runs (30d): 23
  Weekly load: 45.2 km
  Fatigue level: normal
  
  Suggested Pace Range for tempo:
     5:15 - 5:30 min/km

Fetching weather forecast...
  Temperature: 15.6°C
  Precipitation: 0.0 mm/h
  Wind: 11.0 km/h

ML Model Prediction:
  Target Tempo: 165 BPM
  Target Energy: 0.75
  Target Valence: 0.65

Searching for tracks...
  Found 221 unique tracks

Creating Playlist: "Tempo Run | 5:15/km | 10.0km @ 165 BPM"

SUCCESS! Playlist URL: https://open.spotify.com/playlist/...
```

### Update Model (Periodic)

As you log more runs, retrain for improved predictions:

```bash
python src/fetch_strava.py
python src/fetch_weather.py
python src/feature_engineer.py
python src/train_music_model.py
```

Recommended frequency: Monthly, or after significant training changes.

---

## Project Structure

```
runsound-ai/
├── src/
│   ├── fetch_strava.py              # Strava API integration
│   ├── fetch_weather.py             # Weather data collection
│   ├── feature_engineer.py          # Feature engineering
│   ├── train_music_model.py         # Model training
│   ├── ml_recommender.py            # Playlist generation
│   └── spotify_client.py            # Spotify API client
├── data/
│   ├── latest_runs.json             # Cached Strava data
│   ├── run_weather.json             # Weather-matched runs
│   ├── ml_featured_runs.json        # Engineered features
│   ├── latest_playlist.json         # Last generated playlist
│   └── models/
│       └── music_recommender_model.pkl
├── tests/
│   ├── test_spotify_api.py
│   └── test_audio_features.py
├── .env                             # API credentials (not committed)
├── requirements.txt
└── README.md
```

---

## How It Works

### Feature Engineering

Each run is transformed into a feature vector:

```python
{
    "avg_pace_min_km": 5.62,
    "distance_km": 10.5,
    "temp_c": 15.6,
    "temp_bin": "Mild",
    "time_of_day": "Morning",
    "run_type": "tempo",
    "run_length_bin": "Medium",
    "weekly_mileage_km": 45.2,
    "pace_consistency": 0.3,
    "elevation_gain_m": 125.0,
    "precipitation": 0.0,
    "windspeed_kmh": 11.0,
    "humidity": 65.0
}
```

### ML Prediction

Random Forest models trained for each target:

```python
# Model inputs
X = [pace, distance, temp, weather, training_load, ...]

# Model outputs
predictions = {
    "target_tempo": 165,      # BPM
    "target_energy": 0.75,    # 0-1
    "target_valence": 0.65    # 0-1
}
```

### Playlist Generation

Based on predictions:
1. Generates context-aware Spotify search queries
2. Filters by preferred genres
3. Removes duplicates
4. For long runs: structures tracks progressively
5. Creates and populates Spotify playlist

Example workflow:
```
Input:    Tempo, 12km, 5:10/km, 15°C, Morning
Analysis: Recent tempo avg 5:20/km, 45km this week
Output:   165 BPM, 0.75 Energy, 0.65 Valence
Result:   "Tempo Run | 5:10/km | 12.0km @ 165 BPM" playlist
```

---

## Troubleshooting

**"ML model not found"**

Train the model:
```bash
python src/train_music_model.py
```

**"Not enough training data"**

Minimum 10-15 runs required. Model improves with 30+ runs.

**"Spotify 404 error"**

The recommendations endpoint has regional restrictions. The system uses an alternative search-based approach that works globally.

**"Spotify 403 error on audio features"**

Audio features endpoint also has regional restrictions. The system functions without it using search filtering.

**"Strava token expired"**

Refresh tokens expire after 6 months of inactivity. Reauthorize at https://www.strava.com/settings/api and update `.env`.

**"Weather API failed"**

Open-Meteo is free and generally reliable. System uses fallback defaults (15°C, clear) if unavailable.

---

## Model Performance

Performance improves with more training data:

| Runs | Tempo MAE | Energy MAE | Valence MAE |
|------|-----------|------------|-------------|
| 10-15 | ~15 BPM | ~0.12 | ~0.15 |
| 30-40 | ~12 BPM | ~0.10 | ~0.12 |
| 50+ | ~10 BPM | ~0.08 | ~0.10 |

R² scores typically range from 0.65-0.75 after 50+ runs.

The model learns:
- Your typical pace for each run type
- How weather affects your music preferences
- Training load impact on music energy
- Time-of-day patterns
- Progressive vs steady preferences for long runs

---

## Dependencies

**Core:**
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
requests>=2.26.0
python-dotenv>=0.19.0
```

**Optional:**
```
pytest>=6.2.0  # For testing
```

Full list in `requirements.txt`

---

## Known Limitations

**Regional API Restrictions**

Spotify's recommendations and audio-features endpoints return 404/403 in some regions (confirmed in Denmark). The system uses alternative search-based approaches that work globally but may have slightly different results.

**Weather Data**

Historical weather is approximate based on run start time and location. May not reflect exact conditions during the run.

**Training Data**

Model requires minimum 10-15 runs. Performance improves significantly with more data. Cold start problem for new users.

**Playlist Structure**

Progressive playlists for long runs use heuristics, not actual audio feature analysis (due to API limitations in some regions).

---

## Future Development

**Planned Features:**
- Race countdown playlists
- Social features (collaborative filtering)
- Mobile app (React Native)
- Dashboard with visualizations

**Technical Improvements:**
- Model versioning and A/B testing
- Feature importance visualization
- Automated hyperparameter tuning
- Time-of-day circadian optimization
- Cross-user model (cold start solution)

---


