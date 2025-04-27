import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    # Handle missing values
    df = df.fillna(df.mean())
    return df

def feature_engineering(df):
    # Example: Create distance feature from lat/long
    df['distance'] = np.sqrt(df['lat']**2 + df['long']**2)
    return df
