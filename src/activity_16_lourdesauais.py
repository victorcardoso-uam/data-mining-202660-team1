from pathlib import Path

import pandas as pd

SOLAR_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "regression_team1_solar.csv"
solar_data = pd.read_csv(SOLAR_DATA_PATH)

cat > activity16_regression.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("data/regression_team_1_solar.csv")
print(df.shape)
print(df.isna().sum())
print(df.describe().T)

TARGET = "power_output_kw"
df = df.dropna(subset=[TARGET])
print(df.select_dtypes("number").corr()[TARGET].sort_values(ascending=False))

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

results = []

def fit_model(name)