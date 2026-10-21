import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("data/regression_team1_solar.csv")
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

def fit_model(name, features):
    tr = train_df.dropna(subset=features)
    te = test_df.dropna(subset=features)
    X_tr, y_tr = tr[features], tr[TARGET]
    X_te, y_te = te[features], te[TARGET]
    model = LinearRegression().fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    r2_te = r2_score(y_te, pred_te)
    row = {
        "Model": name,
        "Features": ", ".join(features),
        "Train R2": round(r2_score(y_tr, pred_tr), 4),
        "Test R2": round(r2_te, 4),
        "Adj R2": round(adjusted_r2(r2_te, len(y_te), len(features)), 4),
        "MAE": round(mean_absolute_error(y_te, pred_te), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_te, pred_te)), 3),
    }
    results.append(row)
    print(f"\n{name}: intercept={model.intercept_:.3f}, coefs={dict(zip(features, model.coef_.round(4)))}")
    print(row)
    return model, X_te, y_te, pred_te

m1 = fit_model("Model 1", ["solar_irradiance_w_m2"])
m2 = fit_model("Model 2", ["solar_irradiance_w_m2", "panel_temp_c"])
m3 = fit_model("Model 3", ["solar_irradiance_w_m2", "panel_temp_c", "dust_accumulation_pct"])
m4 = fit_model("Model 4", ["solar_irradiance_w_m2", "panel_temp_c", "dust_accumulation_pct", "ambient_temp_c"])
m5 = fit_model("Model 5", ["solar_irradiance_w_m2", "panel_temp_c", "dust_accumulation_pct", "ambient_temp_c", "wind_speed_m_s", "sun_elevation_deg"])

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figures", exist_ok=True)

table = pd.DataFrame(results)
print("\n=== MODEL COMPARISON ===")
print(table.to_string(index=False))
table.to_csv("figures/model_comparison.csv", index=False)

def residual_plots(name, fit):
    model, X_te, y_te, pred = fit
    resid = y_te - pred
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(pred, resid, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set(xlabel="Fitted power (kW)", ylabel="Residual (kW)",
              title=f"{name}: Fitted vs Residuals")
    stats.probplot(resid, dist="norm", plot=ax[1])
    ax[1].set_title(f"{name}: Normal Q-Q")
    fig.tight_layout()
    fname = f"figures/residuals_{name.replace(' ', '_').lower()}.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname} | Shapiro-Wilk p = {stats.shapiro(resid).pvalue:.4f}")

residual_plots("Model 5", m5)
residual_plots("Model 4", m4)
