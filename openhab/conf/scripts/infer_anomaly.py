import argparse
import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "latest_model.joblib")
STATE_PATH = os.path.join(MODELS_DIR, "baseline_state.json")

def compute_feature_zscores(current: dict, means: dict, stds: dict) -> dict:
    z_scores = {}
    for key in ["power", "temp", "humidity"]:
        std = stds.get(key, 1.0) or 1e-6
        z_scores[key] = round((current[key] - means[key]) / std, 2)
    return z_scores

def update_adaptive_threshold(state: dict, new_score: float, is_anomaly: bool):
    recent = state.get("recent_scores", [])
    drift_detected = False

    if not is_anomaly:
        recent.append(new_score)
        if len(recent) > 100:
            recent = recent[-100:]
        state["recent_scores"] = recent

        ewma_mean = float(np.mean(recent))
        ewma_std = float(np.std(recent)) + 1e-4
        new_threshold = ewma_mean + 3.0 * ewma_std

        alpha = 0.15
        adapted = alpha * new_threshold + (1 - alpha) * state["adaptive_threshold"]
        state["adaptive_threshold"] = round(float(adapted), 5)

        if abs(ewma_mean - state["score_mean"]) > 2.0 * state["score_std"]:
            drift_detected = True

    return state["adaptive_threshold"], drift_detected

def classify_severity(score: float, threshold: float) -> str:
    if score < threshold:
        return "NORMAL"
    ratio = score / max(threshold, 1e-5)
    if ratio < 1.25:
        return "WARNING"
    elif ratio < 1.60:
        return "HIGH"
    return "CRITICAL"

def run_inference(power: float, temp: float, humidity: float):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    now = datetime.now()
    hour_float = now.hour + now.minute / 60.0

    input_df = pd.DataFrame([{
        "power": power,
        "temp": temp,
        "humidity": humidity,
        "hour_sin": np.sin(2 * np.pi * hour_float / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_float / 24.0)
    }])

    X_scaled = scaler.transform(input_df)
    raw_score = float(-model.decision_function(X_scaled)[0] + 0.5)
    raw_score = round(max(0.0, raw_score), 4)

    current_threshold = state["adaptive_threshold"]
    is_anomaly = raw_score > current_threshold

    updated_threshold, drift_detected = update_adaptive_threshold(state, raw_score, is_anomaly)
    severity = classify_severity(raw_score, current_threshold)

    z_scores = compute_feature_zscores(
        {"power": power, "temp": temp, "humidity": humidity},
        state["feature_means"],
        state["feature_stds"]
    )

    top_feature = max(z_scores, key=lambda k: abs(z_scores[k]))
    if is_anomaly:
        explanation = f"Аномалія ({severity}): основний вклад {top_feature} (Z={z_scores[top_feature]}), score={raw_score} > {current_threshold}"
    else:
        explanation = f"Нормальний стан (max Z: {top_feature}={z_scores[top_feature]})"

    if drift_detected:
        explanation += " [УВАГА: виявлено дрейф даних, рекомендовано перенавчання]"

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    result = {
        "timestamp": now.isoformat(),
        "model_version": state["version"],
        "anomaly_score": raw_score,
        "threshold": current_threshold,
        "is_anomaly": is_anomaly,
        "severity": severity,
        "z_scores": z_scores,
        "drift_detected": drift_detected,
        "explanation": explanation
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI Anomaly Inference")
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--temp", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    args = parser.parse_args()
    run_inference(args.power, args.temp, args.humidity)
