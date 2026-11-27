# Data Mining & Modern AI Systems (IIND4417) · Team 1

> ## 🎬 Final Capstone Project: **PrestigePick**
> **Everything for the final exam is in [`capstone_streamlit_app/`](capstone_streamlit_app/).**
> Client: HBO Max UX & Content Team · Target: `high_engagement_flag` (Award Caliber carousel) ·
> Models: Decision Tree, Logistic Regression, ANN · Streamlit dashboard with an LLM AI Assistant.
>
> The other files in this repository are earlier course activities, kept exactly as submitted
> because they are graded separately. They are **not** part of the final project (see the map below).

**Team:** Lourdes Auais Bulnes · Matheo Durand · Michel Jakob Häring · Luis Carlos Martínez Máynez ·
Mario Pavón Fuentes · Diego Emilio Sánchez Olvera

## Run the final project

```bash
cd capstone_streamlit_app
python3 -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add an LLM API key for the AI Assistant
streamlit run app.py
```

Full documentation, architecture and test instructions: [`capstone_streamlit_app/README.md`](capstone_streamlit_app/README.md).

## Repository map

### Final capstone (PrestigePick)

| Path | What it is |
|---|---|
| `capstone_streamlit_app/app.py` | **The final Streamlit app**: 4 executive tabs + AI Assistant tab |
| `capstone_streamlit_app/llm_assistant.py` | LLM chatbot: grounding facts, catalog tools, guardrails |
| `capstone_streamlit_app/pipeline.py`, `train_models.py` | Feature pipeline and offline training |
| `capstone_streamlit_app/models/` | Serialized `.joblib` models, scaler, metrics, feature importance |
| `capstone_streamlit_app/data/` | HBO Max catalog + People awards dataset |
| `capstone_streamlit_app/tests/` | 37 offline tests + LLM red-team evaluation |
| `capstone_streamlit_app/requirements.txt` | Dependencies of the final app |
| `src/movie_viewer_clusters.py`, `outputs/movie_clusters/` | Milestone 2 clustering study the capstone personas come from |

### Earlier course activities (graded separately, not part of the final project)

| Activity | Files |
|---|---|
| Activity 13 · Cost-complexity pruning | `src/activity_13_pruning.py`, `src/team_collaboration.py`, `data/processed/pruning_team1_solar.csv` |
| Activity 16 · Regression comparison | `src/activity16_regression.py`, `src/activity_16_lourdesauais.py`, `data/raw/regression_team1_solar.csv`, `figures/activity16_regression/` |
| Activity 20 · Streamlit onboarding (Session 24 template) | `capstone_streamlit_app/app_session24_demo.py`, `capstone_streamlit_app/data/session24_unified_enterprise_demo.csv` |
| Activity 21 · ANN modeling | `activity21_team1_ann.py`, `src/activity21_team1_ann.py`, `data/raw/ann_team1_solar.csv`, `outputs/ann_scaler_team1.joblib`, `outputs/best_ann_model_team1.joblib`, `outputs/benchmark_team1.csv`, `outputs/summary_team1.json`, `figures/ann_diagnostics_team1.png` |
| Partial Exam 2 · Solar regression, ANN and what-if app | `app.py` (repository root), `models/` (repository root), `src/exam2_task1_regression.py`, `src/exam2_task2_ann.py`, `data/raw/exam2_team1_solar.csv`, `outputs/exam2_task1/`, `figures/exam2_task1/`, `figures/exam2_task2/` |

`requirements.txt` at the repository root belongs to these earlier activities; the final app uses
`capstone_streamlit_app/requirements.txt`.
