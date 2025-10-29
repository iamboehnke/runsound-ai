"""
train_music_model.py
Train ML models to predict optimal music features (tempo, energy, valence) 
based on run characteristics.

Models:
- RandomForest for each target (tempo, energy, valence)
- Can be swapped with more sophisticated models later
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "ml_featured_runs.json"

# Features to use for prediction
NUMERIC_FEATURES = [
    "distance_km",
    "avg_pace_min_km",
    "temp_c",
    "precipitation",
    "windspeed_kmh",
    "humidity",
    "elevation_gain_m",
    "pace_consistency",
    "weekly_mileage_km",
]

CATEGORICAL_FEATURES = [
    "time_of_day",
    "temp_bin",
    "run_length_bin",
    "run_type",
]

TARGET_FEATURES = ["target_tempo", "target_energy", "target_valence"]


def load_and_prepare_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data and prepare for training."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run enhanced_feature_engineer.py first."
        )
    
    runs = json.loads(FEATURES_PATH.read_text())
    df = pd.DataFrame(runs)
    
    print(f"Loaded {len(df)} runs")
    print(f"Date range: {df['start_time_utc'].min()} to {df['start_time_utc'].max()}")
    
    # Handle missing values
    df = df.fillna({
        "avg_hr": df["avg_hr"].median() if "avg_hr" in df else 0,
        "humidity": 50,
        "precipitation": 0,
        "windspeed_kmh": 0,
        "elevation_gain_m": 0,
    })
    
    # Encode categorical features
    encoders = {}
    for cat_feature in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[f"{cat_feature}_encoded"] = le.fit_transform(df[cat_feature])
        encoders[cat_feature] = le
    
    return df, encoders


def train_model_for_target(
    X: np.ndarray, 
    y: np.ndarray, 
    target_name: str
) -> RandomForestRegressor:
    """Train a RandomForest model for a specific target."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n--- {target_name} Model Performance ---")
    print(f"  Train MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
    print(f"  Test MAE:  {test_mae:.3f} | R²: {test_r2:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"  CV MAE: {-cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Feature importance
    feature_names = NUMERIC_FEATURES + [f"{f}_encoded" for f in CATEGORICAL_FEATURES]
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]
    
    print(f"  Top Features:")
    for feat, importance in top_features:
        print(f"    {feat}: {importance:.3f}")
    
    return model


def train_all_models():
    """Train models for all target features."""
    
    print("\n" + "="*60)
    print("TRAINING ML MODELS FOR MUSIC RECOMMENDATIONS")
    print("="*60)
    
    # Load data
    df, encoders = load_and_prepare_data()
    
    # Prepare feature matrix
    feature_cols = NUMERIC_FEATURES + [f"{f}_encoded" for f in CATEGORICAL_FEATURES]
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    models = {}
    for target in TARGET_FEATURES:
        y = df[target].values
        model = train_model_for_target(X_scaled, y, target)
        models[target] = model
    
    # Save models and preprocessing objects
    model_artifacts = {
        "models": models,
        "scaler": scaler,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    
    model_path = MODELS_DIR / "music_recommender_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_artifacts, f)
    
    print("\n" + "="*60)
    print(f"Models saved to: {model_path}")
    print("="*60)
    
    return model_artifacts


def predict_music_features(
    model_artifacts: Dict[str, Any],
    run_features: Dict[str, Any]
) -> Dict[str, float]:
    """
    Predict optimal music features for a given run.
    
    Args:
        model_artifacts: Trained models and preprocessing objects
        run_features: Dictionary with run characteristics
        
    Returns:
        Predicted music features (tempo, energy, valence)
    """
    # Extract features
    feature_values = []
    
    # Numeric features
    for feat in model_artifacts["numeric_features"]:
        feature_values.append(run_features.get(feat, 0))
    
    # Categorical features (encoded)
    for feat in model_artifacts["categorical_features"]:
        value = run_features.get(feat, "Unknown")
        encoder = model_artifacts["encoders"][feat]
        # Handle unknown categories
        try:
            encoded = encoder.transform([value])[0]
        except ValueError:
            encoded = 0  # Default to first class
        feature_values.append(encoded)
    
    # Scale
    X = np.array(feature_values).reshape(1, -1)
    X_scaled = model_artifacts["scaler"].transform(X)
    
    # Predict
    predictions = {}
    for target in TARGET_FEATURES:
        model = model_artifacts["models"][target]
        pred = model.predict(X_scaled)[0]
        
        # Clip predictions to valid ranges
        if target == "target_tempo":
            pred = np.clip(pred, 100, 200)
        else:  # energy and valence
            pred = np.clip(pred, 0, 1)
        
        predictions[target] = round(float(pred), 2)
    
    return predictions


if __name__ == "__main__":
    try:
        # Train models
        artifacts = train_all_models()
        
        # Test prediction with latest run
        print("\n" + "="*60)
        print("TESTING PREDICTIONS")
        print("="*60)
        
        runs = json.loads(FEATURES_PATH.read_text())
        test_run = runs[0]  # Latest run
        
        print(f"\nTest Run: {test_run['name']}")
        print(f"  Pace: {test_run['avg_pace_min_km']:.2f} min/km")
        print(f"  Distance: {test_run['distance_km']:.1f} km")
        print(f"  Type: {test_run['run_type']}")
        print(f"  Temp: {test_run['temp_c']:.1f}°C")
        
        predicted = predict_music_features(artifacts, test_run)
        actual = {
            "target_tempo": test_run["target_tempo"],
            "target_energy": test_run["target_energy"],
            "target_valence": test_run["target_valence"],
        }
        
        print("\n  Predictions vs Actual:")
        for key in TARGET_FEATURES:
            print(f"    {key}: {predicted[key]} (actual: {actual[key]})")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run enhanced_feature_engineer.py first.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()