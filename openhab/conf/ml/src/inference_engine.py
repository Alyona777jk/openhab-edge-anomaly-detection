import sys
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from adaptive_threshold import AdaptiveThresholdManager

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/anomaly_if_latest.joblib")
ALERT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../models/alert_dedup.json")

def check_deduplication(alert_key: str, cooldown_sec: int = 900) -> bool:
    """Повертає True, якщо алерт потрібно придушити (дублікат протягом 15 хв)."""
    now = time.time()
    cache = {}
    if os.path.exists(ALERT_CACHE_PATH):
        with open(ALERT_CACHE_PATH, "r") as f:
            cache = json.load(f)

    last_sent = cache.get(alert_key, 0)
    if now - last_sent < cooldown_sec:
        return True

    cache[alert_key] = now
    with open(ALERT_CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return False

def classify_severity(anomaly_score: float, max_z: float) -> str:
    """Класифікація рівня критичності аномалії (0..100 score)."""
    if anomaly_score >= 80 or max_z >= 4.5:
        return "CRITICAL"
    elif anomaly_score >= 55 or max_z >= 3.0:
        return "WARNING"
    return "NORMAL"

def predict(temp: float, humidity: float, power: float):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["features"]

    now = datetime.now()
    hour_float = round(now.hour + now.minute / 60.0, 2)
    
    input_df = pd.DataFrame([{
        "hour": hour_float,
        "temperature": temp,
        "humidity": humidity,
        "power_consumption": power
    }])[features]

    X_scaled = scaler.transform(input_df)
    raw_score = float(model.decision_function(X_scaled)[0])  # Чим менше, тим аномальніше

    # Перетворення raw_score в інтуїтивний Anomaly Score [0..100]
    norm_score = int(np.clip((0.15 - raw_score) / 0.35 * 100, 0, 100))

    # Контекстний Z-score та адаптивний поріг
    thresh_mgr = AdaptiveThresholdManager()
    z_scores = thresh_mgr.compute_contextual_z_scores(
        now.hour, {"temperature": temp, "humidity": humidity, "power_consumption": power}
    )
    max_z_feat = max(z_scores, key=lambda k: abs(z_scores[k]))
    max_z_val = abs(z_scores[max_z_feat])

    is_if_anomaly = raw_score < thresh_mgr.state["dynamic_if_threshold"]
    is_z_anomaly = max_z_val > thresh_mgr.base_z_threshold
    is_anomaly = bool(is_if_anomaly or is_z_anomaly)

    dyn_thresh, drift_detected = thresh_mgr.update_and_check_drift(raw_score, is_anomaly)
    severity = classify_severity(norm_score, max_z_val) if is_anomaly else "NORMAL"

    # Explanation Interface: формування причини відхилення
    reasons = [f"{k} (Z={v:+.1f})" for k, v in z_scores.items() if abs(v) > 2.0]
    if not reasons and is_anomaly:
        reasons.append("Multivariate correlation shift detected by Isolation Forest")

    explanation = "; ".join(reasons) if reasons else "All parameters within seasonal baseline"
    if drift_detected:
        explanation += " [DRIFT DETECTED: Threshold auto-adapted]"

    suppressed = False
    if is_anomaly:
        suppressed = check_deduplication(f"{severity}_{max_z_feat}")

    result = {
        "timestamp": now.isoformat(),
        "model_version": artifact["version"],
        "is_anomaly": is_anomaly,
        "anomaly_score": norm_score,
        "raw_if_score": round(raw_score, 4),
        "dynamic_threshold": dyn_thresh,
        "severity": severity,
        "suppressed_duplicate": suppressed,
        "z_scores": z_scores,
        "explanation": explanation
    }
    print(json.dumps(result))

if __name__ == "__main__":
    # Аргументи командного рядка: temp humidity power
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 21.5
    h = float(sys.argv[2]) if len(sys.argv) > 2 else 47.0
    p = float(sys.argv[3]) if len(sys.argv) > 3 else 155.0
    predict(t, h, p)
