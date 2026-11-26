import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from data_collector import generate_synthetic_baseline, DATA_PATH
from adaptive_threshold import AdaptiveThresholdManager

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def evaluate_model(model, scaler, df_clean):
    """Оцінка Precision та Recall на валідаційній вибірці з ін'єкцією аномалій."""
    features = ["hour", "temperature", "humidity", "power_consumption"]
    X_normal = df_clean[features].copy()
    y_normal = np.zeros(len(X_normal))

    # Створення 50 тестових аномалій (перегрів, стрибок потужності, падіння вологості)
    X_anom = X_normal.sample(50, random_state=42).copy()
    X_anom["temperature"] += np.random.uniform(6.0, 12.0, size=50)
    X_anom["power_consumption"] += np.random.uniform(250.0, 500.0, size=50)
    y_anom = np.ones(len(X_anom))

    X_test = pd.concat([X_normal, X_anom], ignore_index=True)
    y_true = np.concatenate([y_normal, y_anom])

    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)
    # IsolationForest повертає -1 для аномалії, 1 для норми
    y_pred = np.where(preds == -1, 1, 0)

    return {
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4)
    }

def train():
    if not os.path.exists(DATA_PATH):
        df = generate_synthetic_baseline()
    else:
        df = pd.read_csv(DATA_PATH)

    features = ["hour", "temperature", "humidity", "power_consumption"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Edge-оптимізований Isolation Forest (50 дерев для швидкого інференсу < 10 мс)
    model = IsolationForest(
        n_estimators=50,
        max_samples=256,
        contamination=0.02,
        random_state=42
    )
    model.fit(X_scaled)

    # Побудова сезонних профілів для адаптивних порогів
    threshold_mgr = AdaptiveThresholdManager()
    threshold_mgr.fit_seasonal_profiles(df)

    # Валідація моделі
    metrics = evaluate_model(model, scaler, df)
    version = datetime.now().strftime("v%Y%m%d_%H%M")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifact = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "features": features,
        "scaler": scaler,
        "model": model,
        "metrics": metrics
    }

    versioned_path = os.path.join(MODELS_DIR, f"anomaly_if_{version}.joblib")
    latest_path = os.path.join(MODELS_DIR, "anomaly_if_latest.joblib")
    
    joblib.dump(artifact, versioned_path)
    joblib.dump(artifact, latest_path)

    with open(os.path.join(MODELS_DIR, "model_registry.json"), "w") as f:
        json.dump({"active_version": version, "metrics": metrics}, f, indent=2)

    print(f"[SUCCESS] Модель {version} навчена та збережена.")
    print(f"[METRICS] Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1_score']}")

if __name__ == "__main__":
    train()
