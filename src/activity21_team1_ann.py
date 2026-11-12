import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("ann_team1_solar.csv")
target = "power_output_kw"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

common = dict(solver="adam", max_iter=2000, random_state=42)
models = {
    "Model 1 (Shallow Baseline)": MLPRegressor(hidden_layer_sizes=(16,), activation="relu", **common),
    "Model 2 (Single Wide Layer)": MLPRegressor(hidden_layer_sizes=(64,), activation="relu", **common),
    "Model 3 (Deep 2-Layer Net)": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", **common),
    "Model 4 (Deep Tanh)": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="tanh", **common),
    "Model 5 (Regularized Early-Stop)": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                                    alpha=0.01, early_stopping=True, **common),
}

def n_params(m):
    return sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)

n, p = X_test.shape
rows = []
for name, m in models.items():
    m.fit(X_train_s, y_train)
    pr_tr = m.predict(X_train_s)
    pr_te = m.predict(X_test_s)
    r2_te = r2_score(y_test, pr_te)
    rows.append({
        "Model": name,
        "Params": n_params(m),
        "Epochs": m.n_iter_,
        "Train R2": r2_score(y_train, pr_tr),
        "Test R2": r2_te,
        "Adj R2": 1 - (1 - r2_te) * (n - 1) / (n - p - 1),
        "MAE (kW)": mean_absolute_error(y_test, pr_te),
        "RMSE (kW)": np.sqrt(mean_squared_error(y_test, pr_te)),
    })

results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))
results.to_csv("benchmark_team1.csv", index=False)

best_idx = results["Test R2"].idxmax()
best_name = results.loc[best_idx, "Model"]
best = models[best_name]
print("\nWinner:", best_name)

unscaled = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", **common).fit(X_train, y_train)
unscaled_r2 = r2_score(y_test, unscaled.predict(X_test))
print("Model 3 unscaled test R2:", round(unscaled_r2, 4), "epochs:", unscaled.n_iter_)

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
for name, m in models.items():
    a.plot(m.loss_curve_, label=name, lw=1.6)
a.set_yscale("log"); a.set_xlabel("Epoch"); a.set_ylabel("Training loss (squared error / 2, log scale)")
a.set_title("Panel A - Training loss convergence"); a.legend(fontsize=8); a.grid(alpha=.3)

pred = best.predict(X_test_s)
b.scatter(y_test, pred, s=18, alpha=.7, color="#2b6cb0", edgecolor="white", lw=.4)
lim = [0, max(y_test.max(), pred.max()) * 1.05]
b.plot(lim, lim, "--", color="#c53030", lw=1.2, label="Perfect prediction")
b.set_xlim(lim); b.set_ylim(lim)
b.set_xlabel("Actual power output (kW)"); b.set_ylabel("Predicted power output (kW)")
b.set_title(f"Panel B - Predicted vs Actual: {best_name}\nTest R2 = {results.loc[best_idx,'Test R2']:.4f}")
b.legend(); b.grid(alpha=.3)
plt.tight_layout(); plt.savefig("ann_diagnostics_team1.png", dpi=200)

joblib.dump(best, "best_ann_model_team1.joblib")
joblib.dump(scaler, "ann_scaler_team1.joblib")

json.dump({"best": best_name, "unscaled_r2": unscaled_r2, "unscaled_epochs": unscaled.n_iter_,
           "n_train": len(X_train), "n_test": len(X_test), "features": list(X.columns),
           "y_mean": float(y.mean()), "y_std": float(y.std()), "y_min": float(y.min()), "y_max": float(y.max())},
          open("summary_team1.json", "w"), indent=2)
