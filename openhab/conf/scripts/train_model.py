import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "ml", "data", "sensor_history.csv")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

def generate_synthetic_baseline(filepath: str, rows: int = 2000):
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(), periods=rows, freq="5min")
    hours = timestamps.hour + timestamps.minute / 60.0
    diurnal = np.sin(2 * np.pi * (hours - 8) / 24.0)
    power = 120.0 + 35.0 * diurnal + np.random.normal(0, 5.0, rows)
    temp = 22.0 + 1.8 * diurnal + np.random.normal(0, 0.3, rows)
    humidity = 45.0 - 3.0 * diurnal + np.random.normal(0, 1.2, rows)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "power": power,
        "temp": temp,
        "humidity": humidity
    })
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["timestamp"])
    hour_float = dt.dt.hour + dt.dt.minute / 60.0
    features = pd.DataFrame()
    features["power"] = df["power"]
    features["temp"] = df["temp"]
    features["humidity"] = df["humidity"]
    features["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    return features

def train():
    if not os.path.exists(DATA_PATH):
        print("[INFO] Історичні дані не знайдено. Генерація базового датасету...")
        df = generate_synthetic_baseline(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    X = extract_features(df)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    raw_scores = -model.decision_function(X_scaled) + 0.5
    baseline_stats = {
        "version": datetime.now().strftime("v%Y%m%d_%H%M%S"),
        "trained_at": datetime.now().isoformat(),
        "samples_count": len(df),
        "feature_means": df[["power", "temp", "humidity"]].mean().to_dict(),
        "feature_stds": df[["power", "temp", "humidity"]].std().to_dict(),
        "score_mean": float(np.mean(raw_scores)),
        "score_std": float(np.std(raw_scores)),
        "adaptive_threshold": float(np.percentile(raw_scores, 98)),
        "recent_scores": raw_scores[-50:].tolist()
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    version_tag = baseline_stats["version"]
    joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODELS_DIR, f"iforest_{version_tag}.joblib"))
    joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODELS_DIR, "latest_model.joblib"))

    with open(os.path.join(MODELS_DIR, "baseline_state.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"[SUCCESS] Модель {version_tag} навчена. Початковий поріг: {baseline_stats['adaptive_threshold']:.4f}")

if __name__ == "__main__":
    train()
