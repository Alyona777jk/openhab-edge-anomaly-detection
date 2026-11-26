import os
import requests
import pandas as pd

OPENHAB_URL = os.getenv("OPENHAB_URL", "http://localhost:8080/rest/persistence/items")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "ml", "data", "sensor_history.csv")

ITEMS = {
    "power": "Device_Power_Watts",
    "temp": "Room_Temperature",
    "humidity": "Room_Humidity"
}

def fetch_item_history(item_name: str) -> pd.DataFrame:
    resp = requests.get(f"{OPENHAB_URL}/{item_name}", timeout=5)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms").dt.floor("min")
    df["value"] = df["state"].astype(float)
    return df[["timestamp", "value"]].drop_duplicates("timestamp")

def main():
    merged = None
    for col, item in ITEMS.items():
        df = fetch_item_history(item).rename(columns={"value": col})
        merged = df if merged is None else pd.merge(merged, df, on="timestamp", how="inner")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"[OK] Збережено {len(merged)} записів у {OUT_PATH}")

if __name__ == "__main__":
    main()
