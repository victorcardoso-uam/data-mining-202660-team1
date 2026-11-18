"""
Exam 2 - Task 2: Artificial Neural Network (ANN) Training & Loss Convergence
Team 1 | Dataset: data/raw/exam2_team1_solar.csv | Target (y): power_output_kw

- MLPRegressor: 2 hidden layers (64, 32), ReLU, early_stopping=True
- Inputs standardized with StandardScaler (fit on TRAIN only -> no leakage)
- Plots training loss convergence (loss_curve_) across epochs
- Serializes:  models/best_model.joblib   (trained MLP weights)
               models/scaler.joblib       (StandardScaler mean_/scale_)

Run from the repo root:   python3 src/exam2_task2_team1_ann.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Paths & config
# ---------------------------------------------------------------------------
DATA = Path("data/raw/exam2_team1_solar.csv")
MODELS = Path("models")
FIG = Path("figures/exam2_task2")
for d in (MODELS, FIG):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "power_output_kw"
SEED = 42

# ---------------------------------------------------------------------------
# 1. Load, split 80/20 (same split as Task 1), scale
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on train only
X_test_s = scaler.transform(X_test)

print("=" * 78)
print("EXAM 2 - TASK 2 | Team 1 Solar | ANN (MLPRegressor) training & convergence")
print("=" * 78)
print(f"Train: {X_train.shape} | Test: {X_test.shape} | random_state={SEED}")

# ---------------------------------------------------------------------------
# 2. ANN: (64, 32), ReLU, early stopping
# ---------------------------------------------------------------------------
ann = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,               # L2 penalty
    early_stopping=True,        # holds out 10% of TRAIN as validation set
    validation_fraction=0.10,
    n_iter_no_change=20,        # patience: stop after 20 epochs w/o val improvement
    max_iter=5000,
    random_state=SEED,
)
ann.fit(X_train_s, y_train)

n_params = sum(w.size for w in ann.coefs_) + sum(b.size for b in ann.intercepts_)
best_epoch = int(np.argmax(ann.validation_scores_)) + 1
print("\nArchitecture : 6 inputs -> 64 (ReLU) -> 32 (ReLU) -> 1 output (identity)")
print(f"Trainable parameters : {n_params}")
print(f"Epochs run           : {ann.n_iter_}  (early stopping triggered)")
print(f"Best val epoch       : {best_epoch}  (val R2 = {ann.best_validation_score_:.4f})")
print(f"Final training loss  : {ann.loss_curve_[-1]:.4f}  (squared error / 2)")

# ---------------------------------------------------------------------------
# 3. Holdout evaluation
# ---------------------------------------------------------------------------
pred_tr = ann.predict(X_train_s)
pred_te = ann.predict(X_test_s)
n, p = X_test.shape
r2 = r2_score(y_test, pred_te)
metrics = {
    "Train R2": r2_score(y_train, pred_tr),
    "Test R2": r2,
    "Test Adj R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
    "Test MAE (kW)": mean_absolute_error(y_test, pred_te),
    "Test RMSE (kW)": float(np.sqrt(mean_squared_error(y_test, pred_te))),
}
print("\nHOLDOUT TEST-SET METRICS")
for k, v in metrics.items():
    print(f"  {k:16s} {v:.4f}")

# ---------------------------------------------------------------------------
# 4. Loss convergence plot
# ---------------------------------------------------------------------------
epochs = np.arange(1, len(ann.loss_curve_) + 1)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5.2))

a.plot(epochs, ann.loss_curve_, color="#2b6cb0", lw=2, label="Training loss")
a.axvline(best_epoch, color="#c53030", ls="--", lw=1.2, label=f"Best val epoch = {best_epoch}")
a.set_yscale("log")
a.set_xlabel("Epoch"); a.set_ylabel("Loss (squared error / 2, log scale)")
a.set_title("A. Training loss convergence (loss_curve_)")
a.legend(); a.grid(alpha=.3, which="both")

b.plot(epochs, ann.validation_scores_, color="#2f855a", lw=2)
b.axvline(best_epoch, color="#c53030", ls="--", lw=1.2)
b.set_ylim(max(-0.5, min(ann.validation_scores_)), 1.0)
b.set_xlabel("Epoch"); b.set_ylabel("Validation R2")
b.set_title(f"B. Validation R2 (early stopping)\nbest = {ann.best_validation_score_:.4f}")
b.grid(alpha=.3)

c.scatter(y_test, pred_te, s=18, alpha=.7, color="#2b6cb0", edgecolor="white", lw=.4)
lim = [0, max(y_test.max(), pred_te.max()) * 1.05]
c.plot(lim, lim, "--", color="#c53030", lw=1.2, label="Perfect prediction")
c.set_xlim(lim); c.set_ylim(lim)
c.set_xlabel("Actual power output (kW)"); c.set_ylabel("Predicted power output (kW)")
c.set_title(f"C. Test set: predicted vs actual\nR2 = {r2:.4f} | MAE = {metrics['Test MAE (kW)']:.2f} kW")
c.legend(); c.grid(alpha=.3)

fig.suptitle("MLPRegressor (64, 32) ReLU, early_stopping=True | Team 1 Solar",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .94])
plt.savefig(FIG / "ann_loss_convergence.png", dpi=180)
plt.close()
print(f"\nSaved figure -> {FIG / 'ann_loss_convergence.png'}")

# ---------------------------------------------------------------------------
# 5. Serialize model + scaler
# ---------------------------------------------------------------------------
joblib.dump(ann, MODELS / "best_model.joblib")
joblib.dump(scaler, MODELS / "scaler.joblib")
with (MODELS / "model_metadata.json").open("w") as metadata_file:
    json.dump({
        "model": "MLPRegressor(64, 32, relu, early_stopping=True)",
        "features": list(X.columns), "target": TARGET,
        "epochs": ann.n_iter_, "best_val_epoch": best_epoch, "n_params": int(n_params),
        "metrics": metrics,
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X.columns},
        "feature_medians": {c: float(X[c].median()) for c in X.columns},
    }, metadata_file, indent=2)

# Reload check: prove the serialized artifacts reproduce the predictions
m2, s2 = joblib.load(MODELS / "best_model.joblib"), joblib.load(MODELS / "scaler.joblib")
same = np.allclose(m2.predict(s2.transform(X_test)), pred_te)
print(f"Saved model  -> {MODELS / 'best_model.joblib'}")
print(f"Saved scaler -> {MODELS / 'scaler.joblib'}  (mean_ & scale_ for {len(s2.mean_)} features)")
print(f"Saved meta   -> {MODELS / 'model_metadata.json'}")
print(f"Reload check : predictions identical after reload = {same}")
print("=" * 78)