"""
PrestigePick - Viewer Persona Intelligence Dashboard (Capstone Milestone 3/3)
Team 1 | Data Mining & Modern AI Systems (IIND4417)
Client: HBO Max UX & Content Team

Run:  streamlit run app.py
Models are trained offline by train_models.py and loaded here from models/*.joblib.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from pipeline import (AWARD_CALIBER_PERSONA, MODELS_DIR, NUMERIC_FEATURES,
                      PERSONA_POLICY, RATING_MAP, VIEWER_PERSONAS, build_features,
                      load_raw, scale_features, scenario_to_features)

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(page_title="PrestigePick | Viewer Persona Engine", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.1rem; font-weight: 700; margin-bottom: 0.1rem;}
    .sub-header {font-size: 1.0rem; color: #B9AFD0; margin-bottom: 1.2rem;}
    div[data-testid="stMetric"] {background-color: #1C1628; border: 1px solid #3A2F52;
                                 border-radius: 10px; padding: 14px 16px;}
</style>
""", unsafe_allow_html=True)

PERSONA_COLORS = {
    "Mature Drama Night Viewer": "#F87171",
    "Classic Film Buff": "#FBBF24",
    "Easy-Watch Comfort Viewer": "#34D399",
    "Real-Stories Documentary Fan": "#60A5FA",
    "Auteur Cinema Fan": "#F472B6",
    "Prestige Awards Chaser": "#A78BFA",
}
PERSONA_ORDER = [VIEWER_PERSONAS[i] for i in range(len(VIEWER_PERSONAS))]
PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)")
RATING_LABELS = list(RATING_MAP.keys())


# =============================================================================
# 2. CACHED DATA LOADING & SERIALIZED MODEL ARTIFACTS
# =============================================================================
@st.cache_data
def load_data():
    """Raw CSVs + engineered features (same pipeline used for training)."""
    raw_movies, people = load_raw()
    movies, X, genre_cols, params = build_features(raw_movies, people)
    raw_movie_rows = raw_movies[raw_movies["type"] == "Movie"].copy()
    return raw_movie_rows, movies, X, genre_cols, params


@st.cache_resource
def load_artifacts():
    """Load the pre-trained .joblib models, scaler and metrics (no retraining)."""
    meta = joblib.load(MODELS_DIR / "pipeline_metadata.joblib")
    scaler = joblib.load(MODELS_DIR / "standard_scaler.joblib")
    kmeans = joblib.load(MODELS_DIR / "kmeans_personas.joblib")
    classifiers = {name: joblib.load(MODELS_DIR / fname)
                   for name, fname in meta["model_files"].items()}
    with open(MODELS_DIR / "benchmark_metrics.json") as f:
        metrics = json.load(f)
    return meta, scaler, kmeans, classifiers, metrics


try:
    raw_movies, movies, X, genre_cols, params = load_data()
    meta, scaler, kmeans, classifiers, metrics = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run `python train_models.py` inside "
             "capstone_streamlit_app/ first, then reload this page.")
    st.stop()

X_scaled = scale_features(X, scaler, genre_cols)
movies["cluster"] = kmeans.predict(X_scaled)
movies["viewer_profile"] = movies["cluster"].map(VIEWER_PERSONAS)
X_test = X_scaled.loc[meta["test_index"]]
y_test = movies.loc[meta["test_index"], "cluster"]
perf_df = pd.DataFrame(metrics["classifiers"])
GENRES = [g.replace("g_", "") for g in genre_cols]


# =============================================================================
# 3. SESSION STATE (What-If presets + scenario log)
# =============================================================================
PRESETS = {
    "Oscar-winning epic": dict(imdb=8.4, rotten=94, year=2003, rating="PG-13/TV-14",
                               genres=["Drama", "Adventure"], title_aw=6, actor_aw=4,
                               director_aw=3, platforms=2),
    "Weekend family comedy": dict(imdb=5.6, rotten=35, year=2008, rating="PG/TV-PG",
                                  genres=["Comedy", "Family"], title_aw=0, actor_aw=0,
                                  director_aw=0, platforms=2),
    "True-crime documentary": dict(imdb=7.3, rotten=88, year=2018, rating="PG-13/TV-14",
                                   genres=["Documentary", "Crime"], title_aw=0, actor_aw=0,
                                   director_aw=0, platforms=1),
    "1950s black-and-white classic": dict(imdb=7.8, rotten=95, year=1954, rating="PG/TV-PG",
                                          genres=["Drama", "Romance"], title_aw=0, actor_aw=0,
                                          director_aw=0, platforms=1),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[f"wi_{k}"] = v


if "wi_imdb" not in st.session_state:
    apply_preset("Oscar-winning epic")
if "scenario_log" not in st.session_state:
    st.session_state.scenario_log = []


# =============================================================================
# 4. SIDEBAR
# =============================================================================
st.sidebar.title("🎬 PrestigePick")
st.sidebar.caption("Team 1 · Data Mining & Modern AI Systems (IIND4417)")
st.sidebar.divider()
st.sidebar.subheader("🎛️ Inference settings")
model_name = st.sidebar.selectbox("Active persona classifier", list(classifiers.keys()),
                                  index=2, key="active_model")
threshold = st.sidebar.slider("Award Caliber cutoff", 0.05, 0.95, 0.50, 0.05,
                              help="Minimum probability of the 'Prestige Awards Chaser' "
                                   "persona needed to auto-curate a title into the carousel.")
st.sidebar.divider()
st.sidebar.markdown(
    "**Client:** HBO Max UX & Content Team  \n"
    "**Unsupervised:** K-Means (k=6) viewer personas  \n"
    "**Supervised:** Decision Tree · Logistic Regression · ANN (MLP)  \n"
    "**Data:** HBO Max movie catalog + People awards dataset")

active_model = classifiers[model_name]
active_row = perf_df[perf_df["Model"] == model_name].iloc[0]


# =============================================================================
# 5. HEADER & KPI CARDS
# =============================================================================
st.markdown('<div class="main-header">🎬 PrestigePick · Viewer Persona Intelligence Engine</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clustering the HBO Max movie catalog into viewer personas '
            'and auto-curating the <b>Award Caliber</b> carousel</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Movies analyzed", f"{len(movies):,}", delta=f"{len(raw_movies) - len(movies)} dropped (no genre/IMDb)",
          delta_color="off")
k2.metric("Viewer personas", f"{metrics['kmeans']['k']} clusters",
          delta=f"Silhouette {metrics['kmeans']['silhouette']:.3f}", delta_color="off")
award_share = (movies["viewer_profile"] == AWARD_CALIBER_PERSONA).mean()
k3.metric("Award Caliber titles", f"{award_share:.1%}",
          delta=f"{(movies['viewer_profile'] == AWARD_CALIBER_PERSONA).sum()} movies", delta_color="off")
k4.metric("Active model test F1", f"{active_row['F1 (macro)']:.3f}",
          delta=model_name.split(" (")[0], delta_color="off")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer & EDA",
    "⚙️ Preprocessing & Correlation",
    "🤖 Multi-Model Benchmark",
    "🔮 Real-Time What-If Inference",
])

# -----------------------------------------------------------------------------
# TAB 1: DATA EXPLORER & EDA
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("HBO Max movie catalog")
    c1, c2 = st.columns([3, 2])
    with c1:
        persona_filter = st.multiselect("Filter by viewer persona", PERSONA_ORDER, key="eda_filter")
        view = movies if not persona_filter else movies[movies["viewer_profile"].isin(persona_filter)]
        st.dataframe(view[["title", "year", "rating", "Genre", "imdb_score", "rotten_score",
                           "title_awards", "director_awards", "viewer_profile"]],
                     width="stretch", height=330, hide_index=True)
    with c2:
        st.write("**Summary statistics (engineered features)**")
        stats_cols = ["imdb_score", "rotten_score", "year", "maturity", "title_awards",
                      "actor_awards", "director_awards", "n_platforms"]
        st.dataframe(movies[stats_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(2),
                     width="stretch")

    st.divider()
    c3, c4 = st.columns([3, 2])
    with c3:
        feat = st.selectbox("Feature distribution", stats_cols, key="eda_feat")
        fig = px.histogram(movies, x=feat, color="viewer_profile", barmode="overlay", opacity=0.6,
                           nbins=40, category_orders={"viewer_profile": PERSONA_ORDER},
                           color_discrete_map=PERSONA_COLORS,
                           title=f"{feat} by viewer persona")
        fig.update_layout(**PLOT_LAYOUT, legend_title_text="Persona")
        st.plotly_chart(fig, width="stretch")
    with c4:
        sizes = movies["viewer_profile"].value_counts().reindex(PERSONA_ORDER).reset_index()
        sizes.columns = ["Persona", "Movies"]
        fig = px.bar(sizes, x="Movies", y="Persona", orientation="h", color="Persona",
                     color_discrete_map=PERSONA_COLORS, title="Movies per persona")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: PREPROCESSING & CORRELATION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Missing value treatment")
    raw = raw_movies.dropna(subset=["Genre", "imdb_score"])
    miss = pd.DataFrame({
        "Missing before": raw[["rotten_score", "rating"]].isna().sum(),
        "Treatment": [f"Linear fit from IMDb: RT = {params['rotten_slope']:.1f} × IMDb "
                      f"{params['rotten_intercept']:+.1f} (+ missing flag)",
                      f"Ordinal median ({RATING_LABELS[int(params['maturity_median'])]}) + missing flag"],
        "Missing after": [0, 0],
    })
    st.dataframe(miss, width="stretch")
    st.caption("Rows without a genre or IMDb score were dropped "
               f"({len(raw_movies) - len(movies)} of {len(raw_movies)} movies).")

    st.subheader("Encoding & feature engineering")
    st.markdown(
        "- **Maturity rating** → ordinal 0–3 (G → R)\n"
        f"- **Genres** → multi-hot, {len(genre_cols)} genres kept (≥15 movies each), weighted ×1.5\n"
        "- **Awards won by the title** (Oscar, Golden Globe, BAFTA, Emmy) → count, log1p\n"
        "- **Star power** → awards held by the cast and directors (People dataset), log1p\n"
        "- **Movie age** = 2026 − year; **platforms** → count, log1p")

    st.subheader("StandardScaler: before vs after")
    sc_feat = st.selectbox("Numeric feature", NUMERIC_FEATURES, key="scale_feat")
    cb, ca = st.columns(2)
    with cb:
        fig = px.histogram(X, x=sc_feat, nbins=40, title=f"Raw {sc_feat}",
                           color_discrete_sequence=["#FBBF24"])
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with ca:
        fig = px.histogram(X_scaled, x=sc_feat, nbins=40, title=f"Scaled {sc_feat} (mean 0, std 1)",
                           color_discrete_sequence=["#A78BFA"])
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Pearson correlation matrix")
    corr_cols = ["imdb_score", "rotten_score", "maturity", "year", "title_awards",
                 "actor_awards", "director_awards", "n_platforms"]
    fig = px.imshow(movies[corr_cols].corr(), text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(**PLOT_LAYOUT, height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("IMDb and Rotten Tomatoes move together, which is why the missing Rotten scores "
               "can be estimated from IMDb.")

# -----------------------------------------------------------------------------
# TAB 3: MULTI-MODEL BENCHMARK
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Benchmark: classical models vs Neural Network")
    st.caption("Task: assign each movie to its viewer persona. Metrics are on the 20% held-out "
               "test set (315 movies); train accuracy and 5-fold CV are shown to check overfitting.")
    table = perf_df.copy()
    table["Silhouette"] = "—"
    table["R²"] = "n/a (classification)"
    km = metrics["kmeans"]
    table = pd.concat([table, pd.DataFrame([{
        "Model": "K-Means (k=6)", "Family": "Unsupervised", "Silhouette": f"{km['silhouette']:.3f}",
        "R²": "n/a (clustering)"}])], ignore_index=True)
    num_cols = ["Train Accuracy", "Test Accuracy", "Precision (macro)", "Recall (macro)",
                "F1 (macro)", "ROC-AUC (OvR)", "5-fold CV F1"]
    for c in num_cols:
        table[c] = table[c].apply(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(f"K-Means stability: mean ARI {km['stability_mean_ari']:.3f} across 10 random seeds "
               "(1.0 = identical clusters every time).")

    st.subheader("Confusion matrices (test set)")
    short = [p.split()[0] + " " + p.split()[1] for p in PERSONA_ORDER]
    cm_cols = st.columns(3)
    for col, (name, model) in zip(cm_cols, classifiers.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), labels=range(len(PERSONA_ORDER)))
        fig = px.imshow(cm, text_auto=True, x=short, y=short, color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual"), title=name)
        fig.update_layout(**PLOT_LAYOUT, height=420, coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=50, b=10))
        col.plotly_chart(fig, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("ROC curves: Award Caliber vs rest")
        award_id = PERSONA_ORDER.index(AWARD_CALIBER_PERSONA)
        y_bin = (y_test == award_id).astype(int)
        fig = go.Figure()
        for name, model in classifiers.items():
            p = model.predict_proba(X_test)[:, award_id]
            fpr, tpr, _ = roc_curve(y_bin, p)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{name} (AUC={roc_auc_score(y_bin, p):.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="False positive rate",
                          yaxis_title="True positive rate", legend=dict(y=0.05, x=0.4))
        st.plotly_chart(fig, width="stretch")
    with c6:
        st.subheader("Neural Network loss convergence")
        ann = classifiers["Neural Network (MLP)"]
        epochs = np.arange(1, len(ann.loss_curve_) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=ann.loss_curve_, name="Training loss",
                                 line=dict(color="#A78BFA")))
        fig.add_trace(go.Scatter(x=epochs, y=ann.validation_scores_, name="Validation accuracy",
                                 yaxis="y2", line=dict(color="#34D399")))
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="Epoch", yaxis_title="Cross-entropy loss",
                          yaxis2=dict(title="Validation accuracy", overlaying="y", side="right",
                                      range=[0, 1]), legend=dict(y=0.5, x=0.55))
        st.plotly_chart(fig, width="stretch")
        best_ep = int(np.argmax(ann.validation_scores_)) + 1
        st.caption(f"Loss falls quickly in the first epochs and then flattens. Validation accuracy "
                   f"peaks at epoch {best_ep} ({max(ann.validation_scores_):.3f}); early stopping "
                   f"ended training {len(epochs) - best_ep} epochs later, so the saved weights are "
                   "not overfitted.")

    st.subheader("Persona map (PCA projection of the K-Means clusters)")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                           "Persona": movies["viewer_profile"], "Title": movies["title"]})
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Persona", hover_name="Title",
                     color_discrete_map=PERSONA_COLORS, opacity=0.75,
                     category_orders={"Persona": PERSONA_ORDER},
                     title=f"{pca.explained_variance_ratio_.sum():.0%} of variance shown in 2D")
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(**PLOT_LAYOUT, height=520)
    st.plotly_chart(fig, width="stretch")

# -----------------------------------------------------------------------------
# TAB 4: REAL-TIME WHAT-IF SCENARIO INFERENCE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Should this title go into the Award Caliber carousel?")
    st.write("Describe a movie and the loaded model places it in a viewer persona in real time.")

    p_cols = st.columns(len(PRESETS))
    for col, name in zip(p_cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### 🎛️ Title attributes")
        imdb = st.slider("IMDb score", 1.0, 10.0, step=0.1, key="wi_imdb")
        rotten = st.slider("Rotten Tomatoes score", 0, 100, key="wi_rotten")
        title_aw = st.slider("Major awards won by the title (Oscar, Globe, BAFTA, Emmy)", 0, 15,
                             key="wi_title_aw")
        director_aw = st.slider("Awards held by the director(s)", 0, 15, key="wi_director_aw")
        actor_aw = st.slider("Awards held by the cast", 0, 25, key="wi_actor_aw")
        year = st.slider("Release year", 1920, 2026, key="wi_year")
        rating = st.selectbox("Maturity rating", RATING_LABELS, key="wi_rating")
        genres = st.multiselect("Genres", GENRES, key="wi_genres")
        platforms = st.slider("Number of streaming platforms", 1, 6, key="wi_platforms")

    feat_row = scenario_to_features(imdb, rotten, year, rating, genres, title_aw, actor_aw,
                                    director_aw, platforms, genre_cols)
    feat_scaled = scale_features(feat_row, scaler, genre_cols)
    probs = active_model.predict_proba(feat_scaled)[0]
    pred_id = int(np.argmax(probs))
    pred_persona = VIEWER_PERSONAS[pred_id]
    award_prob = float(probs[PERSONA_ORDER.index(AWARD_CALIBER_PERSONA)])
    kmeans_persona = VIEWER_PERSONAS[int(kmeans.predict(feat_scaled)[0])]

    with right:
        st.markdown(f"##### 🎯 Live inference · {model_name}")
        st.markdown(f"**Award Caliber probability: {award_prob:.1%}** (cutoff {threshold:.2f})")
        st.progress(award_prob)

        if award_prob >= threshold:
            st.success(f"✅ AWARD CALIBER: add this title to the 'Award Caliber' carousel "
                       f"(probability {award_prob:.1%} ≥ {threshold:.2f}).")
        else:
            st.error(f"⛔ NOT AWARD CALIBER: keep it out of the 'Award Caliber' carousel "
                     f"(probability {award_prob:.1%} < {threshold:.2f}).")
        st.markdown(f"**Predicted persona:** {pred_persona}  \n"
                    f"**Recommended placement:** {PERSONA_POLICY[pred_persona]}")
        if kmeans_persona != pred_persona:
            st.caption(f"K-Means nearest centroid says: {kmeans_persona} (borderline title).")

        prob_df = pd.DataFrame({"Persona": PERSONA_ORDER, "Probability": probs})
        fig = px.bar(prob_df, x="Probability", y="Persona", orientation="h", color="Persona",
                     color_discrete_map=PERSONA_COLORS, range_x=[0, 1], text_auto=".1%")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, height=300, yaxis_title=None,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        similar = (movies[movies["viewer_profile"] == pred_persona]
                   .sort_values("imdb_score", ascending=False)
                   [["title", "year", "imdb_score"]].head(5))
        st.write(f"**Top-rated titles already in this persona:**")
        st.dataframe(similar, hide_index=True, width="stretch")

    st.divider()
    if st.button("📌 Save this scenario to the log"):
        st.session_state.scenario_log.append({
            "IMDb": imdb, "RT": rotten, "Title awards": title_aw, "Director awards": director_aw,
            "Year": year, "Genres": ", ".join(genres), "Model": model_name.split(" (")[0],
            "Persona": pred_persona, "Award Caliber %": round(award_prob * 100, 1),
            "Decision": "Carousel" if award_prob >= threshold else "Standard row"})
    if st.session_state.scenario_log:
        st.write("**Scenario log (this session)**")
        st.dataframe(pd.DataFrame(st.session_state.scenario_log), hide_index=True,
                     width="stretch")
