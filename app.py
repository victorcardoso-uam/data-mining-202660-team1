import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "exam2_team1_solar.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, scaler, metadata


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.set_page_config(
    page_title="Team 1 Solar - Exam 2",
    page_icon="SUN",
    layout="wide",
)

st.title("Solar Power Operations Dashboard")
st.caption("Exam 2 | Team 1 | Serialized ANN and standardized feature pipeline")

try:
    model, scaler, metadata = load_artifacts()
    df = load_data()
except FileNotFoundError as exc:
    st.error(f"Missing serialized artifact: {exc.filename}")
    st.stop()

features = metadata["features"]
target = metadata["target"]
tabs = st.tabs([
    "Dataset & Data Quality",
    "Feature Ranges",
    "ANN Model Benchmark",
    "Real-Time What-If Inference",
])

with tabs[0]:
    st.subheader("Assigned solar dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", len(features))
    c3.metric("Missing values", int(df.isna().sum().sum()))
    st.dataframe(df, use_container_width=True, height=360)

with tabs[1]:
    st.subheader("Feature ranges from the training data")
    ranges = metadata["feature_ranges"]
    range_df = pd.DataFrame(
        {
            "Feature": features,
            "Minimum": [ranges[name][0] for name in features],
            "Maximum": [ranges[name][1] for name in features],
            "Median": [metadata["feature_medians"][name] for name in features],
        }
    )
    st.dataframe(range_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Serialized MLPRegressor performance")
    metrics = metadata["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R2", f"{metrics['Test R2']:.4f}")
    c2.metric("Test adjusted R2", f"{metrics['Test Adj R2']:.4f}")
    c3.metric("Test MAE", f"{metrics['Test MAE (kW)']:.2f} kW")
    c4.metric("Test RMSE", f"{metrics['Test RMSE (kW)']:.2f} kW")
    st.json(metadata)
    st.success(
        f"Loaded {MODEL_PATH.name} and {SCALER_PATH.name}; "
        f"the model has {metadata['n_params']} trainable parameters."
    )

with tabs[3]:
    st.subheader("Real-Time What-If Scenario")
    col1, col2 = st.columns(2)
    ranges = metadata["feature_ranges"]

    def slider(name, label, step):
        low, high = ranges[name]
        default = metadata["feature_medians"][name]
        return col1.slider(
            label,
            min_value=float(low),
            max_value=float(high),
            value=float(default),
            step=step,
        )

    irradiance = slider("solar_irradiance_w_m2", "Solar irradiance (W/m2)", 10.0)
    ambient = slider("ambient_temp_c", "Ambient temperature (C)", 0.5)
    panel = slider("panel_temp_c", "Panel temperature (C)", 0.5)
    wind = slider("wind_speed_m_s", "Wind speed (m/s)", 0.1)
    dust = slider("dust_accumulation_pct", "Dust accumulation (%)", 0.5)
    efficiency = slider("inverter_efficiency", "Inverter efficiency", 0.005)

    input_df = pd.DataFrame([{
        "solar_irradiance_w_m2": irradiance,
        "ambient_temp_c": ambient,
        "panel_temp_c": panel,
        "wind_speed_m_s": wind,
        "dust_accumulation_pct": dust,
        "inverter_efficiency": efficiency,
    }])[features]

    with col2:
        prediction = float(model.predict(scaler.transform(input_df))[0])
        st.metric("Predicted power output", f"{prediction:.2f} kW")

        high_risk = (
            prediction < 250
            or dust >= 30
            or panel >= 65
            or efficiency < 0.90
        )
        if high_risk:
            st.error("HIGH-RISK OPERATIONAL ALERT")
            st.write(
                "Recommended actions: clean panels, inspect inverter efficiency, "
                "and verify high-temperature operating conditions."
            )
        else:
            st.success("NORMAL OPERATIONAL STATUS")
            st.write("No high-risk policy condition was triggered.")
