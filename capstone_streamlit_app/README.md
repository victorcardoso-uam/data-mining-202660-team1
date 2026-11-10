# PrestigePick · Viewer Persona Intelligence Engine (Team 1, Milestone 3/3)

Streamlit dashboard for the HBO Max UX & Content Team. Movies are clustered with
K-Means (k = 6) into viewer personas; Decision Tree, Logistic Regression and a
Neural Network (MLP) are trained to assign new titles to a persona. The
**Award Caliber** carousel takes titles whose probability of the
"Prestige Awards Chaser" persona is above the cutoff in the sidebar.

## Run

```bash
cd capstone_streamlit_app
python3 -m pip install -r requirements.txt
python3 train_models.py      # builds models/*.joblib (run again after changing the data or pipeline)
streamlit run app.py
```

## Layout

```
capstone_streamlit_app/
├── app.py                 # 4-tab dashboard, loads the .joblib models (no training)
├── pipeline.py            # shared cleaning + feature engineering
├── train_models.py        # trains and saves the models + benchmark metrics
├── data/                  # HBO_MAX_Content_extended.csv, People_Dataset.csv
├── models/                # kmeans, scaler, decision tree, logistic, ANN (.joblib) + metrics
├── .streamlit/config.toml # dark theme
└── requirements.txt
```

The features and K-Means settings are the same as `src/movie_viewer_clusters.py`,
so the six clusters match the Milestone 2 personas exactly.
