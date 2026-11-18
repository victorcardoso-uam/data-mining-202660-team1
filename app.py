"""
SolarSense - Inverter Risk & Power Output What-If Dashboard
Exam 2 - Task 3 | Team 1 | Data Mining
Target: power_output_kw | Dataset: data/raw/exam2_team1_solar.csv

Run from the repo root:   streamlit run app.py
Loads serialized artifacts produced by Tasks 1 & 2:
  models/best_model.joblib            (MLPRegressor 64-32 ReLU)
  models/scaler.joblib                (StandardScaler)
  models/model_metadata.json
  outputs/exam2_task1/task1_best_model.joblib   (Task 1 best OLS model, optional)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. CONFIG
# =============================================================================
st.set_page_config(page_title="SolarSense | Team 1", page_icon="☀️", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data/raw/exam2_team1_solar.csv"
MODEL_PATH = ROOT / "models/best_model.joblib"
SCALER_PATH = ROOT / "models/scaler.joblib"
META_PATH = ROOT / "models/model_metadata.json"
OLS_PATH = ROOT / "outputs/exam2_task1/task1_best_model.joblib"
TARGET = "power_output_kw"
SEED = 42

LABELS = {
    "solar_irradiance_w_m2": "Solar irradiance (W/m²)",
    "ambient_temp_c": "Ambient temperature (°C)",
    "panel_temp_c": "Panel temperature (°C)",
    "wind_speed_m_s": "Wind speed (m/s)",
    "dust_accumulation_pct": "Dust accumulation (%)",
    "inverter_efficiency": "Inverter efficiency (0-1)",
}
PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))


# =============================================================================
# 2. CACHED LOADERS (artifacts are loaded once and shared by all 4 tabs)
# =============================================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = json.load(open(META_PATH))
    ols = joblib.load(OLS_PATH) if OLS_PATH.exists() else None
    return model, scaler, meta, ols


@st.cache_resource
def panel_temp_model(df):
    """Physics-style sub-model: panel temp rises with irradiance & ambient temp,
    falls with wind (NOCT-type relation). Lets the what-if sliders stay realistic."""
    cols = ["solar_irradiance_w_m2", "ambient_temp_c", "wind_speed_m_s"]
    lr = LinearRegression().fit(df[cols], df["panel_temp_c"])
    return lr, cols, r2_score(df["panel_temp_c"], lr.predict(df[cols]))


try:
    df = load_data()
    ann, scaler, meta, ols_bundle = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}. Run `python3 src/exam2_task1_regression.py` and "
             "`python3 src/exam2_task2_ann.py` from the repo root first.")
    st.stop()

FEATURES = meta["features"]
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


def predict_ann(frame):
    return ann.predict(scaler.transform(frame[FEATURES]))


# =============================================================================
# 3. SIDEBAR: artifact status + operational policy thresholds
# =============================================================================
with st.sidebar:
    st.header("☀️ SolarSense")
    st.caption("Team 1 · Exam 2 · Data Mining")
    st.subheader("Loaded artifacts")
    st.success(f"✅ best_model.joblib\n\n{type(ann).__name__} {ann.hidden_layer_sizes}, "
               f"{ann.activation}, {ann.n_iter_} epochs")
    st.success(f"✅ scaler.joblib\n\n{type(scaler).__name__}, {len(scaler.mean_)} features")
    if ols_bundle:
        st.success(f"✅ task1_best_model.joblib\n\n{ols_bundle['name']}")
    else:
        st.warning("Task 1 OLS artifact not found (optional)")

    st.subheader("Inverter alert policy")
    RATED_KW = st.number_input("Inverter rated capacity (kW)", 300, 800, 550, 10)
    LOAD_WARN = st.slider("Load warning (% of rated)", 60, 100, 85)
    LOAD_CRIT = st.slider("Overload / clipping alert (% of rated)", 70, 110, 95)
    PANEL_WARN = st.slider("Panel temp warning (°C)", 45, 80, 60)
    PANEL_CRIT = st.slider("Panel temp critical (°C)", 50, 90, 68)
    AMB_CRIT = st.slider("Ambient temp derating limit (°C)", 30, 50, 40)

# =============================================================================
# 4. HEADER + TABS
# =============================================================================
st.title("☀️ SolarSense: Power Output & Inverter Risk Dashboard")
st.caption("Predicts **power_output_kw** from plant conditions with a serialized MLP "
           "(Task 2) and flags inverter operating risk in real time.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Data Overview", "📈 2. Model Performance",
    "🧠 3. ANN Diagnostics", "⚡ 4. What-If Scenario Inference"])

# ----------------------------------------------------------------------------- TAB 1
with tab1:
    st.subheader("Dataset overview")
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Features", len(FEATURES))
    c[2].metric("Mean output", f"{y.mean():.1f} kW")
    c[3].metric("Peak output", f"{y.max():.1f} kW")

    # Model-in-the-loop: ANN fitted values over the whole dataset
    view = df.copy()
    view["predicted_kw (ANN)"] = predict_ann(view)
    st.dataframe(view.head(15), width="stretch")

    left, right = st.columns(2)
    corr = df.corr()[TARGET].drop(TARGET).sort_values()
    left.plotly_chart(px.bar(corr, orientation="h", title="Correlation with power_output_kw",
                             labels={"value": "Pearson r", "index": ""}).update_layout(
        showlegend=False, **PLOT), width="stretch")
    feat = right.selectbox("Feature vs output", FEATURES, format_func=LABELS.get)
    right.plotly_chart(px.scatter(view, x=feat, y=TARGET, color="panel_temp_c",
                                  color_continuous_scale="Turbo", opacity=.7,
                                  labels=LABELS, title=f"{LABELS[feat]} vs output").update_layout(**PLOT),
                       width="stretch")

# ----------------------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Holdout performance of the serialized models (20% test set)")

    def metrics(name, pred, p):
        r2 = r2_score(y_test, pred)
        n = len(y_test)
        return {"Model": name, "R²": r2, "Adj R²": 1 - (1 - r2) * (n - 1) / (n - p - 1),
                "MAE (kW)": mean_absolute_error(y_test, pred),
                "RMSE (kW)": np.sqrt(mean_squared_error(y_test, pred))}

    preds = {"ANN MLP (64, 32)": predict_ann(X_test)}
    rows = [metrics("ANN MLP (64, 32)", preds["ANN MLP (64, 32)"], len(FEATURES))]
    if ols_bundle:
        preds[ols_bundle["name"]] = ols_bundle["model"].predict(X_test)
        rows.append(metrics(ols_bundle["name"], preds[ols_bundle["name"]], len(FEATURES)))
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.style.format("{:.4f}").highlight_max(subset=["R²", "Adj R²"], color="#c6f6d5")
                 .highlight_min(subset=["MAE (kW)", "RMSE (kW)"], color="#c6f6d5"),
                 width="stretch")

    left, right = st.columns(2)
    fig = go.Figure()
    for name, pr in preds.items():
        fig.add_scatter(x=y_test, y=pr, mode="markers", name=name, opacity=.7)
    lim = [0, float(max(y_test.max(), max(p.max() for p in preds.values())) * 1.05)]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Perfect", line=dict(dash="dash", color="red"))
    fig.update_layout(title="Predicted vs actual (test)", xaxis_title="Actual kW",
                      yaxis_title="Predicted kW", **PLOT)
    left.plotly_chart(fig, width="stretch")

    res = pd.DataFrame({name: y_test.values - pr for name, pr in preds.items()}).melt(
        var_name="Model", value_name="Residual (kW)")
    right.plotly_chart(px.histogram(res, x="Residual (kW)", color="Model", barmode="overlay",
                                    nbins=30, opacity=.6, title="Residual distribution (test)")
                       .update_layout(**PLOT), width="stretch")

# ----------------------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("ANN training diagnostics (read from best_model.joblib)")
    c = st.columns(4)
    n_params = sum(w.size for w in ann.coefs_) + sum(b.size for b in ann.intercepts_)
    c[0].metric("Architecture", f"6 → {' → '.join(map(str, ann.hidden_layer_sizes))} → 1")
    c[1].metric("Trainable params", f"{n_params:,}")
    c[2].metric("Epochs (early stop)", ann.n_iter_)
    c[3].metric("Best validation R²", f"{ann.best_validation_score_:.4f}")

    left, right = st.columns(2)
    ep = np.arange(1, len(ann.loss_curve_) + 1)
    left.plotly_chart(go.Figure(go.Scatter(x=ep, y=ann.loss_curve_, mode="lines"))
                      .update_layout(title="Training loss (loss_curve_)", xaxis_title="Epoch",
                                     yaxis_title="Loss (log)", yaxis_type="log", **PLOT),
                      width="stretch")
    right.plotly_chart(go.Figure(go.Scatter(x=ep, y=ann.validation_scores_, mode="lines",
                                            line=dict(color="green")))
                       .update_layout(title="Validation R² per epoch", xaxis_title="Epoch",
                                      yaxis_title="R²", yaxis_range=[0, 1], **PLOT),
                       width="stretch")

    # Permutation-style sensitivity: how much does test MAE grow if a feature is shuffled?
    rng = np.random.default_rng(SEED)
    base = mean_absolute_error(y_test, predict_ann(X_test))
    imp = {}
    for f in FEATURES:
        Xp = X_test.copy()
        Xp[f] = rng.permutation(Xp[f].values)
        imp[LABELS[f]] = mean_absolute_error(y_test, predict_ann(Xp)) - base
    imp = pd.Series(imp).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h",
                           title="Permutation importance (increase in test MAE, kW)",
                           labels={"value": "Δ MAE (kW)", "index": ""})
                    .update_layout(showlegend=False, **PLOT), width="stretch")

# ----------------------------------------------------------------------------- TAB 4
with tab4:
    st.subheader("Real-time what-if scenario: adjust irradiance & temperature → inverter alert")
    pt_model, pt_cols, pt_r2 = panel_temp_model(df)
    rng_ = meta["feature_ranges"]
    med = meta["feature_medians"]

    st.markdown("**Primary scenario drivers**")
    c1, c2 = st.columns(2)
    irr = c1.slider(LABELS["solar_irradiance_w_m2"], 0.0, 1200.0,
                    float(round(med["solar_irradiance_w_m2"])), 10.0)
    amb = c2.slider(LABELS["ambient_temp_c"], 0.0, 50.0, float(round(med["ambient_temp_c"])), 0.5)

    with st.expander("Secondary conditions", expanded=True):
        c3, c4, c5 = st.columns(3)
        wind = c3.slider(LABELS["wind_speed_m_s"], 0.0, 20.0, float(round(med["wind_speed_m_s"], 1)), 0.1)
        dust = c4.slider(LABELS["dust_accumulation_pct"], 0.0, 50.0,
                         float(round(med["dust_accumulation_pct"], 1)), 0.5)
        eff = c5.slider(LABELS["inverter_efficiency"], 0.80, 1.00,
                        float(round(med["inverter_efficiency"], 3)), 0.001, format="%.3f")
        auto_pt = st.checkbox("Estimate panel temperature from irradiance, ambient temp & wind",
                              value=True, help=f"Linear thermal sub-model, R² = {pt_r2:.3f}")
        est_pt = float(pt_model.predict(pd.DataFrame([[irr, amb, wind]], columns=pt_cols))[0])
        panel = est_pt if auto_pt else st.slider(LABELS["panel_temp_c"], 0.0, 90.0,
                                                 float(round(est_pt, 1)), 0.5)

    scenario = pd.DataFrame([{
        "solar_irradiance_w_m2": irr, "ambient_temp_c": amb, "panel_temp_c": panel,
        "wind_speed_m_s": wind, "dust_accumulation_pct": dust, "inverter_efficiency": eff}])
    pred_kw = float(max(0.0, predict_ann(scenario)[0]))
    load_pct = 100 * pred_kw / RATED_KW

    # ---- operational policy
    reasons_crit, reasons_warn = [], []
    if load_pct >= LOAD_CRIT:
        reasons_crit.append(f"Inverter load {load_pct:.0f}% ≥ {LOAD_CRIT}% of rated → overload / clipping")
    elif load_pct >= LOAD_WARN:
        reasons_warn.append(f"Inverter load {load_pct:.0f}% ≥ {LOAD_WARN}% of rated")
    if panel >= PANEL_CRIT:
        reasons_crit.append(f"Panel temp {panel:.1f} °C ≥ {PANEL_CRIT} °C → thermal derating / hotspot risk")
    elif panel >= PANEL_WARN:
        reasons_warn.append(f"Panel temp {panel:.1f} °C ≥ {PANEL_WARN} °C")
    if amb >= AMB_CRIT:
        reasons_crit.append(f"Ambient {amb:.1f} °C ≥ {AMB_CRIT} °C → inverter cabinet overheating")
    out_of_range = [LABELS[k] for k, v in scenario.iloc[0].items()
                    if not (rng_[k][0] <= v <= rng_[k][1])]

    m = st.columns(4)
    m[0].metric("Predicted power output", f"{pred_kw:.1f} kW")
    m[1].metric("Inverter load", f"{load_pct:.1f} %", f"{pred_kw - RATED_KW:+.1f} kW vs rated",
                delta_color="inverse")
    m[2].metric("Panel temperature", f"{panel:.1f} °C", "estimated" if auto_pt else "manual",
                delta_color="off")
    m[3].metric("Fleet average output", f"{y.mean():.1f} kW", f"{pred_kw - y.mean():+.1f} kW")

    if reasons_crit:
        st.error("🚨 **HIGH-RISK INVERTER ALERT**: curtail output / dispatch maintenance\n\n"
                 + "\n".join(f"- {r}" for r in reasons_crit + reasons_warn))
    elif reasons_warn:
        st.warning("⚠️ **ELEVATED: monitor inverter**\n\n" + "\n".join(f"- {r}" for r in reasons_warn))
    else:
        st.success("✅ **NORMAL OPERATION**: inverter load and temperatures within policy limits")
    if out_of_range:
        st.info("ℹ️ Extrapolation: outside the training range for " + ", ".join(out_of_range)
                + ". Treat the prediction with caution.")

    # Gauge + sensitivity curve
    left, right = st.columns([1, 1.4])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pred_kw, number={"suffix": " kW"},
        title={"text": "Inverter load vs rated capacity"},
        gauge={"axis": {"range": [0, RATED_KW * 1.15]},
               "bar": {"color": "#1a202c"},
               "steps": [{"range": [0, RATED_KW * LOAD_WARN / 100], "color": "#c6f6d5"},
                         {"range": [RATED_KW * LOAD_WARN / 100, RATED_KW * LOAD_CRIT / 100], "color": "#fefcbf"},
                         {"range": [RATED_KW * LOAD_CRIT / 100, RATED_KW * 1.15], "color": "#fed7d7"}],
               "threshold": {"line": {"color": "red", "width": 4}, "value": RATED_KW}}))
    gauge.update_layout(height=330, margin=dict(l=20, r=20, t=60, b=10))
    left.plotly_chart(gauge, width="stretch")

    sweep = pd.DataFrame({"solar_irradiance_w_m2": np.linspace(0, 1200, 121)})
    sweep["ambient_temp_c"], sweep["wind_speed_m_s"] = amb, wind
    sweep["dust_accumulation_pct"], sweep["inverter_efficiency"] = dust, eff
    sweep["panel_temp_c"] = (pt_model.predict(sweep[pt_cols]) if auto_pt else panel)
    sweep["pred"] = np.clip(predict_ann(sweep), 0, None)
    fig = go.Figure()
    fig.add_scatter(x=sweep["solar_irradiance_w_m2"], y=sweep["pred"], mode="lines",
                    name=f"ANN prediction @ {amb:.1f} °C ambient")
    fig.add_hline(y=RATED_KW, line_dash="dash", line_color="red", annotation_text="Rated capacity")
    fig.add_hline(y=RATED_KW * LOAD_CRIT / 100, line_dash="dot", line_color="orange",
                  annotation_text=f"{LOAD_CRIT}% alert")
    fig.add_scatter(x=[irr], y=[pred_kw], mode="markers", name="Current scenario",
                    marker=dict(size=14, color="red", symbol="star"))
    fig.update_layout(title="Sensitivity: output vs irradiance (other sliders fixed)",
                      xaxis_title="Irradiance (W/m²)", yaxis_title="Predicted kW", **PLOT)
    right.plotly_chart(fig, width="stretch")

    with st.expander("Scenario input vector sent to the model"):
        st.dataframe(scenario.T.rename(columns={0: "value"}), width="stretch")
        st.caption("Pipeline: raw inputs → scaler.joblib (StandardScaler) → best_model.joblib (MLP)")
