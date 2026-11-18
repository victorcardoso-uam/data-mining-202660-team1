from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "exam2_team1_solar.csv"
FIG_DIR = REPO_ROOT / "figures" / "exam2_task1_team1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "power_output_kw"
FEATURES = [
    "solar_irradiance_w_m2",
    "ambient_temp_c",
    "panel_temp_c",
    "wind_speed_m_s",
    "dust_accumulation_pct",
    "inverter_efficiency",
]


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def residual_dashboard(name, fitted, residuals, design_matrix):
    standardized = residuals / np.std(residuals, ddof=1)
    sqrt_standardized = np.sqrt(np.abs(standardized))
    design = np.column_stack([np.ones(len(design_matrix)), design_matrix])
    hat = np.einsum(
        "ij,jk,ik->i",
        design,
        np.linalg.pinv(design.T @ design),
        design,
    )
    p = design.shape[1]
    cooks_distance = (
        (standardized**2 / p)
        * (hat / np.maximum((1 - hat)**2, np.finfo(float).eps))
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{name}: Classical OLS Residual Diagnostics", fontsize=14)

    axes[0, 0].scatter(fitted, residuals, alpha=0.65, edgecolor="black", linewidth=0.25)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q")

    axes[1, 0].scatter(fitted, sqrt_standardized, alpha=0.65, edgecolor="black", linewidth=0.25)
    fitted_sorted = np.argsort(fitted)
    axes[1, 0].plot(
        np.asarray(fitted)[fitted_sorted],
        sqrt_standardized[fitted_sorted],
        color="steelblue",
        alpha=0.45,
        linewidth=1,
    )
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("Sqrt(|Standardized residual|)")

    axes[1, 1].scatter(hat, standardized, alpha=0.65, edgecolor="black", linewidth=0.25)
    axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Residuals vs Leverage")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].set_ylabel("Standardized residuals")

    influence_limit = np.quantile(cooks_distance, 0.95)
    if np.isfinite(influence_limit) and influence_limit > 0:
        for level in (0.5, 1.0):
            axes[1, 1].text(
                0.98,
                level,
                f"Cook's D={level}",
                transform=axes[1, 1].transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="darkred",
            )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = FIG_DIR / f"{name.lower().replace(' ', '_').replace('-', '_')}_residual_diagnostics.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved residual dashboard: {output_path}")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])

    if set(FEATURES + [TARGET]).issubset(df.columns) is False:
        raise ValueError("The assigned CSV does not contain the required columns.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}")

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    log_model = LinearRegression()
    log_model.fit(X_train, y_train_log)
    log_pred = np.expm1(log_model.predict(X_test))

    polynomial_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )
    polynomial_model.fit(X_train, y_train)
    polynomial_pred = polynomial_model.predict(X_test)

    models = [
        (
            "Baseline OLS Linear",
            baseline_pred,
            len(FEATURES),
        ),
        (
            "Log-Transformed Target OLS",
            log_pred,
            len(FEATURES),
        ),
        (
            "Polynomial Regression Degree 2",
            polynomial_pred,
            polynomial_model.named_steps["polynomialfeatures"].n_output_features_,
        ),
    ]

    results = []
    for name, prediction, parameter_count in models:
        r2 = r2_score(y_test, prediction)
        results.append(
            {
                "Model": name,
                "R2": r2,
                "Adjusted R2": adjusted_r2(r2, len(y_test), parameter_count),
                "MAE": mean_absolute_error(y_test, prediction),
                "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
            }
        )

    comparison = pd.DataFrame(results)
    comparison_path = FIG_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.6f")
    print("\n=== MODEL COMPARISON (20% HOLDOUT TEST SET) ===")
    print(comparison.round(6).to_string(index=False))
    print(f"Saved comparison table: {comparison_path}")

    for name, prediction, _ in models:
        residuals = y_test.to_numpy() - prediction
        if name == "Polynomial Regression Degree 2":
            design_matrix = polynomial_model.named_steps[
                "polynomialfeatures"
            ].transform(X_test)
        else:
            design_matrix = X_test.to_numpy()
        residual_dashboard(name, prediction, residuals, design_matrix)
        shapiro = stats.shapiro(residuals)
        print(f"{name}: Shapiro-Wilk p-value = {shapiro.pvalue:.6f}")


if __name__ == "__main__":
    main()
