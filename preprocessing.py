import pandas as pd
import numpy as np

def basic_cleaning(df):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['forecast_prob_rain', 'cloud_cover', 'temp_c', 'humidity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col] = 0.0
    if 'actual_rain' in df.columns:
        df['actual_rain'] = pd.to_numeric(df['actual_rain'], errors='coerce').fillna(0).astype(int)
    else:
        df['actual_rain'] = 0
    return df

def add_binary_features(df, cloud_threshold=0.5):
    df = df.copy()
    df['is_cloudy'] = (df['cloud_cover'] >= cloud_threshold).astype(int)
    df['forecast_prob'] = (df['forecast_prob_rain'] / 100.0).clip(0, 1)
    df['forecast_bin'] = pd.cut(df['forecast_prob'], bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
                                labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    return df
