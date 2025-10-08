import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_weather.csv")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def generate_synthetic_data(n_days=365, seed=42):
    np.random.seed(seed)
    start = datetime(2024, 1, 1)
    rows = []
    for i in range(n_days):
        date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        day_frac = i / 365.0
        seasonal = 0.2 + 0.3 * np.sin(2 * np.pi * day_frac)
        base_p = np.clip(seasonal, 0.01, 0.7)
        forecast_prob_rain = np.clip((base_p + np.random.normal(0, 0.08)) * 100, 0, 100)
        cloud_cover = np.clip(np.random.beta(2, 5) * (0.5 + base_p), 0, 1)
        humidity = np.clip(0.4 + 0.6 * base_p + np.random.normal(0, 0.08), 0, 1)
        rain_prob = np.clip(base_p + 0.4 * cloud_cover + 0.2 * (humidity - 0.5), 0, 0.95)
        actual_rain = np.random.rand() < rain_prob
        temp_c = 25 - 10 * seasonal + np.random.normal(0, 3)
        rows.append({
            "date": date,
            "forecast_prob_rain": round(float(forecast_prob_rain), 1),
            "actual_rain": int(actual_rain),
            "cloud_cover": round(float(cloud_cover), 3),
            "temp_c": round(float(temp_c), 2),
            "humidity": round(float(humidity), 3),
        })
    return pd.DataFrame(rows)

def load_or_generate(path=SAMPLE_PATH):
    ensure_data_dir()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"[data_loader] Loaded dataset from {path} with {len(df)} rows.")
            return df
        except Exception:
            pass
    df = generate_synthetic_data()
    df.to_csv(path, index=False)
    print(f"[data_loader] Generated synthetic dataset and saved to {path}")
    return df
