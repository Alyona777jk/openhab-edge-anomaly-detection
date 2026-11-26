import json
import os
import numpy as np

STATE_FILE = os.path.join(os.path.dirname(__file__), "../models/threshold_state.json")

class AdaptiveThresholdManager:
    def __init__(self, alpha: float = 0.05, base_z_threshold: float = 3.0):
        self.alpha = alpha  # Коефіцієнт згладжування EWMA
        self.base_z_threshold = base_z_threshold
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        return {
            "hourly_profiles": {},  # Сезонний профіль для кожної години (0-23)
            "ewma_score_mean": 0.0,
            "ewma_score_var": 0.01,
            "dynamic_if_threshold": -0.05,
            "drift_counter": 0
        }

    def save_state(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def fit_seasonal_profiles(self, df):
        """Побудова статистичного профілю (mean, std) для кожної години доби."""
        df["hour_int"] = df["timestamp"].apply(lambda x: int(str(x)[11:13]))
        features = ["temperature", "humidity", "power_consumption"]
        profiles = {}
        for h in range(24):
            subset = df[df["hour_int"] == h]
            profiles[str(h)] = {
                feat: {
                    "mean": float(subset[feat].mean()),
                    "std": float(max(subset[feat].std(), 0.1))
                } for feat in features
            }
        self.state["hourly_profiles"] = profiles
        self.save_state()

    def compute_contextual_z_scores(self, hour: int, reading: dict) -> dict:
        """Розрахунок сезонного Z-score для кожної метрики."""
        profile = self.state["hourly_profiles"].get(str(hour), {})
        z_scores = {}
        for feat, val in reading.items():
            if feat in profile:
                mean = profile[feat]["mean"]
                std = profile[feat]["std"]
                z_scores[feat] = round((val - mean) / std, 3)
        return z_scores

    def update_and_check_drift(self, raw_if_score: float, is_confirmed_anomaly: bool) -> tuple:
        """
        Оновлення динамічного порогу та перевірка концептуального дрейфу (Concept Drift).
        Якщо нормальні дані систематично зміщуються, поріг адаптується.
        """
        if not is_confirmed_anomaly:
            # Оновлення EWMA для нормальних значень score
            diff = raw_if_score - self.state["ewma_score_mean"]
            self.state["ewma_score_mean"] += self.alpha * diff
            self.state["ewma_score_var"] = (1 - self.alpha) * (
                self.state["ewma_score_var"] + self.alpha * (diff ** 2)
            )
            std_score = np.sqrt(self.state["ewma_score_var"])
            # Динамічний поріг = середнє - 2.5 * стандартне відхилення
            self.state["dynamic_if_threshold"] = round(
                self.state["ewma_score_mean"] - 2.5 * std_score, 4
            )
            self.state["drift_counter"] = 0
        else:
            self.state["drift_counter"] += 1

        # Якщо 12 вимірювань поспіль (3 години) фіксується відхилення — це дрейф режиму роботи
        drift_detected = self.state["drift_counter"] >= 12
        if drift_detected:
            # Адаптація порогу до нового режиму
            self.state["dynamic_if_threshold"] = round(raw_if_score - 0.05, 4)
            self.state["drift_counter"] = 0

        self.save_state()
        return self.state["dynamic_if_threshold"], drift_detected
