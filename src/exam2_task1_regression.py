"""
Exam 2 - Task 1: Regression Model Comparison & Residual Diagnostics
Team 1 | Dataset: data/raw/exam2_team1_solar.csv | Target (y): power_output_kw

Models
  (1) Baseline OLS Linear
  (2) Log-Transformed Target OLS   -> fit on ln(y), back-transformed to kW (Duan smearing)
  (3) Polynomial Regression (degree 2) -> all squares + pairwise interactions, then OLS

Holdout metrics (test set, in kW): R2, Adjusted R2, MAE, RMSE
4-panel OLS residual dashboard (training fit, per model):
  A. Residuals vs Fitted      -> linearity / zero-mean errors
  B. Normal Q-Q               -> normality of residuals
  C. Scale-Location           -> homoscedasticity (constant variance)
  D. Residuals vs Leverage    -> influential points (Cook's distance)

Run from the repo root:   python src/exam2_task1_regression.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ---------------------------------------------------------------------------
# 0. Paths & config
# ---------------------------------------------------------------------------
DATA = Path("data/raw/exam2_team1_solar.csv")
FIG = Path("figures/exam2_task1")
OUT = Path("outputs/exam2_task1")
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "power_output_kw"
SEED = 42

# ---------------------------------------------------------------------------
# 1. Load + 80/20 split
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
print("=" * 78)
print("EXAM 2 - TASK 1 | Team 1 Solar | Regression comparison & residual diagnostics")
print("=" * 78)
print(f"Loaded {DATA}  ->  {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Missing values: {int(df.isna().sum().sum())} | Duplicate rows: {int(df.duplicated().sum())}")

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
print(f"Train: {len(X_train)} rows (80%) | Test: {len(X_test)} rows (20%) | random_state={SEED}")
print(f"Features ({X.shape[1]}): {list(X.columns)}\n")


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
def adjusted_r2(r2, n, p):
    """p = number of predictors the model actually uses (excl. intercept)."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


class LogTargetOLS:
    """OLS on ln(y). Predictions back-transformed with Duan's smearing estimator
    (exp(pred) alone underestimates the mean of y)."""

    def __init__(self):
        self.ols = LinearRegression()

    def fit(self, X, y):
        log_y = np.log(y)
        self.ols.fit(X, log_y)
        resid = log_y - self.ols.predict(X)
        self.smear_ = np.mean(np.exp(resid))
        return self

    def predict_log(self, X):
        return self.ols.predict(X)

    def predict(self, X):
        return np.exp(self.ols.predict(X)) * self.smear_


models = {
    "M1 Baseline OLS": LinearRegression(),
    "M2 Log-Target OLS": LogTargetOLS(),
    "M3 Polynomial (deg 2)": Pipeline([
        ("scaler", StandardScaler()),            # keeps x^2 terms well-conditioned
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ols", LinearRegression()),
    ]),
}

n_test = len(X_test)
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    pred_tr, pred_te = m.predict(X_train), m.predict(X_test)
    if name.startswith("M3"):
        p = m.named_steps["poly"].n_output_features_
    else:
        p = X.shape[1]
    r2 = r2_score(y_test, pred_te)
    rows.append({
        "Model": name,
        "Predictors (p)": p,
        "Train R2": r2_score(y_train, pred_tr),
        "Test R2": r2,
        "Test Adj R2": adjusted_r2(r2, n_test, p),
        "Test MAE (kW)": mean_absolute_error(y_test, pred_te),
        "Test RMSE (kW)": np.sqrt(mean_squared_error(y_test, pred_te)),
    })

results = pd.DataFrame(rows)
best_idx = results["Test Adj R2"].idxmax()
best_name = results.loc[best_idx, "Model"]
results["Winner"] = np.where(results.index == best_idx, "<== best", "")

pd.set_option("display.width", 200)
print("-" * 78)
print("HOLDOUT TEST-SET METRICS (all errors in kW)")
print("-" * 78)
print(results.round(4).to_string(index=False))
print(f"\nBest model by Test Adjusted R2: {best_name}")
results.drop(columns="Winner").to_csv(OUT / "task1_model_comparison.csv", index=False)

# Baseline OLS coefficients (interpretability)
ols = models["M1 Baseline OLS"]
coef = pd.Series(ols.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nBaseline OLS coefficients (kW per unit change):")
print(f"  {'intercept':24s} {ols.intercept_:+.4f}")
for k, v in coef.items():
    print(f"  {k:24s} {v:+.4f}")


# ---------------------------------------------------------------------------
# 3. Residual diagnostics (classical OLS checks on the training fit)
# ---------------------------------------------------------------------------
def design_matrix(name, m, X_):
    """Matrix the OLS step actually sees, with intercept column, for leverage."""
    if name.startswith("M3"):
        Z = m[:-1].transform(X_)
    else:
        Z = np.asarray(X_, dtype=float)
    return np.column_stack([np.ones(len(Z)), Z])


def ols_diagnostics(name, m):
    if name.startswith("M2"):
        y_fit = m.predict_log(X_train)          # diagnostics live in the space OLS was fit in
        y_obs = np.log(y_train.values)
        unit = "ln(kW)"
    else:
        y_fit = m.predict(X_train)
        y_obs = y_train.values
        unit = "kW"
    resid = y_obs - y_fit
    Z = design_matrix(name, m, X_train)
    n, k = Z.shape                                # k = p + 1 (with intercept)
    H = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    lev = np.clip(np.diag(H), 0, 0.9999)
    sigma = np.sqrt(np.sum(resid ** 2) / (n - k))
    std_resid = resid / (sigma * np.sqrt(1 - lev))
    cooks = std_resid ** 2 * lev / (k * (1 - lev))

    # Formal tests to back up the plots
    shapiro_p = stats.shapiro(resid).pvalue
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
    # Breusch-Pagan (Koenker): regress e^2 on the design matrix, LM = n * R2
    e2 = resid ** 2
    beta = np.linalg.lstsq(Z, e2, rcond=None)[0]
    r2_aux = 1 - np.sum((e2 - Z @ beta) ** 2) / np.sum((e2 - e2.mean()) ** 2)
    bp_lm = n * r2_aux
    bp_p = stats.chi2.sf(bp_lm, k - 1)
    cook_thr = 4 / n
    n_infl = int(np.sum(cooks > cook_thr))
    return dict(y_fit=y_fit, resid=resid, std_resid=std_resid, lev=lev, cooks=cooks,
                unit=unit, n=n, k=k, shapiro_p=shapiro_p, dw=dw, bp_lm=bp_lm, bp_p=bp_p,
                cook_thr=cook_thr, n_infl=n_infl)


def lowess_line(x, y, frac=0.4):
    """Light-weight LOWESS-like smoother (moving average on sorted x)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    w = max(5, int(len(xs) * frac / 4))
    smooth = pd.Series(ys).rolling(w, center=True, min_periods=1).mean().values
    return xs, smooth


def dashboard(name, d, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (a, b), (c, e) = axes
    blue, red = "#2b6cb0", "#c53030"

    # A. Residuals vs Fitted
    a.scatter(d["y_fit"], d["resid"], s=14, alpha=.6, color=blue, edgecolor="white", lw=.3)
    a.axhline(0, color="grey", ls="--", lw=1)
    a.plot(*lowess_line(d["y_fit"], d["resid"]), color=red, lw=2, label="smoothed trend")
    a.set_xlabel(f"Fitted values ({d['unit']})"); a.set_ylabel(f"Residuals ({d['unit']})")
    a.set_title("A. Residuals vs Fitted  (linearity)"); a.legend(); a.grid(alpha=.3)

    # B. Normal Q-Q
    (osm, osr), (slope, intercept, _) = stats.probplot(d["std_resid"], dist="norm")
    b.scatter(osm, osr, s=14, alpha=.6, color=blue, edgecolor="white", lw=.3)
    b.plot(osm, slope * osm + intercept, color=red, lw=2)
    b.set_xlabel("Theoretical quantiles"); b.set_ylabel("Standardized residuals")
    b.set_title(f"B. Normal Q-Q  (normality)\nShapiro-Wilk p = {d['shapiro_p']:.4f}")
    b.grid(alpha=.3)

    # C. Scale-Location
    sl = np.sqrt(np.abs(d["std_resid"]))
    c.scatter(d["y_fit"], sl, s=14, alpha=.6, color=blue, edgecolor="white", lw=.3)
    c.plot(*lowess_line(d["y_fit"], sl), color=red, lw=2)
    c.set_xlabel(f"Fitted values ({d['unit']})"); c.set_ylabel("√|Standardized residuals|")
    c.set_title(f"C. Scale-Location  (homoscedasticity)\n"
                f"Breusch-Pagan LM = {d['bp_lm']:.2f}, p = {d['bp_p']:.4f}")
    c.grid(alpha=.3)

    # D. Residuals vs Leverage + Cook's distance contours
    infl = d["cooks"] > d["cook_thr"]
    e.scatter(d["lev"][~infl], d["std_resid"][~infl], s=14, alpha=.6, color=blue,
              edgecolor="white", lw=.3, label="regular")
    e.scatter(d["lev"][infl], d["std_resid"][infl], s=22, alpha=.9, color=red,
              label=f"Cook's D > 4/n ({d['n_infl']} pts)")
    hx = np.linspace(max(d["lev"].min(), 1e-3), d["lev"].max() * 1.05, 200)
    for D, ls in [(0.5, "--"), (1.0, ":")]:
        bound = np.sqrt(D * d["k"] * (1 - hx) / hx)
        e.plot(hx, bound, color="grey", ls=ls, lw=1, label=f"Cook's D = {D}")
        e.plot(hx, -bound, color="grey", ls=ls, lw=1)
    lim = max(4, np.abs(d["std_resid"]).max() * 1.15)
    e.set_ylim(-lim, lim)
    top = np.argsort(d["cooks"])[-3:]
    for i in top:
        e.annotate(str(X_train.index[i]), (d["lev"][i], d["std_resid"][i]), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
    e.axhline(0, color="grey", lw=.8)
    e.set_xlabel("Leverage (hat value)"); e.set_ylabel("Standardized residuals")
    e.set_title("D. Residuals vs Leverage  (influential points)")
    e.legend(fontsize=8, loc="lower right"); e.grid(alpha=.3)

    fig.suptitle(f"Residual Diagnostic Dashboard - {name}  |  Team 1 Solar (train fit, n={d['n']})"
                 f"\nDurbin-Watson = {d['dw']:.3f}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, .95])
    plt.savefig(path, dpi=180)
    plt.close()


print("\n" + "-" * 78)
print("RESIDUAL HEALTH CHECKS (training fit)")
print("-" * 78)
diag_rows = []
for name, m in models.items():
    d = ols_diagnostics(name, m)
    slug = name.split()[0].lower()
    path = FIG / f"residual_dashboard_{slug}.png"
    dashboard(name, d, path)
    verdict = lambda ok: "PASS" if ok else "FLAG"
    diag_rows.append({
        "Model": name,
        "Shapiro-Wilk p (normality)": d["shapiro_p"],
        "Normality": verdict(d["shapiro_p"] > .05),
        "Breusch-Pagan p (const. variance)": d["bp_p"],
        "Homoscedasticity": verdict(d["bp_p"] > .05),
        "Durbin-Watson (independence)": d["dw"],
        "Independence": verdict(1.5 < d["dw"] < 2.5),
        "Influential pts (Cook>4/n)": d["n_infl"],
        "Max Cook's D": d["cooks"].max(),
    })
    print(f"  saved {path}")

diag = pd.DataFrame(diag_rows)
diag.to_csv(OUT / "task1_residual_checks.csv", index=False)
print()
print(diag.round(4).to_string(index=False))

# Copy of the winner's dashboard under a fixed name for the report
best_slug = best_name.split()[0].lower()
(FIG / "residual_dashboard_BEST.png").write_bytes(
    (FIG / f"residual_dashboard_{best_slug}.png").read_bytes())

# Side-by-side predicted vs actual for the 3 models (test set)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, m) in zip(axes, models.items()):
    pr = m.predict(X_test)
    ax.scatter(y_test, pr, s=16, alpha=.7, color="#2b6cb0", edgecolor="white", lw=.3)
    lim = [0, max(y_test.max(), pr.max()) * 1.05]
    ax.plot(lim, lim, "--", color="#c53030"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"{name}\nTest R2 = {r2_score(y_test, pr):.4f}")
    ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted (kW)"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "task1_pred_vs_actual.png", dpi=180); plt.close()

joblib.dump({"model": models[best_name], "name": best_name, "features": list(X.columns)},
            OUT / "task1_best_model.joblib")
print(f"\nSaved: {OUT}/task1_model_comparison.csv, task1_residual_checks.csv, task1_best_model.joblib")
print(f"Figures: {FIG}/residual_dashboard_m1|m2|m3|BEST.png, task1_pred_vs_actual.png")
print("=" * 78)