import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sensor_history.csv")
OPENHAB_URL = "http://localhost:8080/rest/persistence/items"

def fetch_from_openhab(item_name: str) -> pd.DataFrame:
    """Отримання історичних даних з OpenHAB Persistence REST API."""
    try:
        resp = requests.get(f"{OPENHAB_URL}/{item_name}", timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
        df[item_name] = df["state"].astype(float)
        return df[["timestamp", item_name]]
    except Exception as e:
        print(f"[INFO] OpenHAB API недоступний ({e}). Використовується локальний генератор.")
        return pd.DataFrame()

def generate_synthetic_baseline(days: int = 14) -> pd.DataFrame:
    """Генерація базової поведінки сенсорів з урахуванням добової сезонності."""
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(minutes=15 * i) for i in range(days * 96)]
    timestamps.reverse()

    hours = np.array([t.hour + t.minute / 60.0 for t in timestamps])
    
    # Синусоїдальна добова сезонність + нормальний шум
    temp = 21.0 + 2.5 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 0.3, len(timestamps))
    humidity = 48.0 - 5.0 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 1.2, len(timestamps))
    power_w = 150.0 + 80.0 * (np.sin((hours - 7) * np.pi / 12) > 0) + np.random.normal(0, 15.0, len(timestamps))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hours,
        "temperature": np.round(temp, 2),
        "humidity": np.round(humidity, 2),
        "power_consumption": np.round(np.clip(power_w, 20, None), 2)
    })
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[OK] Збережено {len(df)} записів у {DATA_PATH}")
    return df

if __name__ == "__main__":
    generate_synthetic_baseline()
