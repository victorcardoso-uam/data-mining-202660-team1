"""
PrestigePick feature pipeline (shared by train_models.py and app.py)
--------------------------------------------------------------------
Same cleaning and feature engineering as src/movie_viewer_clusters.py,
written as functions so the training script and the Streamlit app build
features in exactly the same way.
"""

from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
MOVIES_PATH = DATA_DIR / "HBO_MAX_Content_extended.csv"
PEOPLE_PATH = DATA_DIR / "People_Dataset.csv"

RANDOM_STATE = 42
K = 6
GENRE_WEIGHT = 1.5
MIN_GENRE_COUNT = 15
CURRENT_YEAR = 2026

RATING_MAP = {"G/TV-G": 0, "PG/TV-PG": 1, "PG-13/TV-14": 2, "R/TV-MA": 3}
AWARD_COLS = ["Academy_Awards", "Golden_Globe_Awards", "BAFTA_Awards", "Emmy_Awards"]

NUMERIC_FEATURES = ["imdb_score", "rotten_score", "maturity", "movie_age",
                    "title_awards_log", "actor_awards_log",
                    "director_awards_log", "n_platforms_log"]

# K-Means cluster id -> viewer persona (same mapping as src/movie_viewer_clusters.py)
VIEWER_PERSONAS = {
    0: "Mature Drama Night Viewer",
    1: "Classic Film Buff",
    2: "Easy-Watch Comfort Viewer",
    3: "Real-Stories Documentary Fan",
    4: "Auteur Cinema Fan",
    5: "Prestige Awards Chaser",
}
AWARD_CALIBER_PERSONA = "Prestige Awards Chaser"

# One carousel / policy per persona for the HBO Max UX & Content Team
PERSONA_POLICY = {
    "Mature Drama Night Viewer": "Place in the late-evening 'Gripping Dramas' row for R-rated drama fans.",
    "Classic Film Buff": "Place in the 'Timeless Classics' row next to the TCM collection.",
    "Easy-Watch Comfort Viewer": "Place in the 'Easy Watch' family and comedy row; keep out of prestige rows.",
    "Real-Stories Documentary Fan": "Place in the 'True Stories' documentary row.",
    "Auteur Cinema Fan": "Place in the 'Director's Spotlight' row, grouped by filmmaker.",
    "Prestige Awards Chaser": "Auto-curate into the 'Award Caliber' carousel on the home page.",
}


def split_list(s, sep=","):
    return [x.strip() for x in str(s).split(sep) if x.strip()] if pd.notna(s) else []


def load_raw():
    movies = pd.read_csv(MOVIES_PATH)
    people = pd.read_csv(PEOPLE_PATH)
    return movies, people


def build_features(movies, people):
    """Clean movies and engineer features.

    Returns (movies_df, X, genre_cols, params) where X has the numeric
    features followed by multi-hot genre columns (unscaled), and params holds
    the values learned from the data that new inputs need too.
    """
    movies = movies[movies["type"] == "Movie"].copy()
    movies = movies.dropna(subset=["Genre", "imdb_score"]).reset_index(drop=True)

    # Rotten score: impute from IMDb with a linear fit, keep a missing flag
    movies["rotten_missing"] = movies["rotten_score"].isna().astype(int)
    known = movies.dropna(subset=["rotten_score"])
    slope, intercept = np.polyfit(known["imdb_score"], known["rotten_score"], 1)
    movies["rotten_score"] = movies["rotten_score"].fillna(
        (movies["imdb_score"] * slope + intercept).clip(0, 100))

    # Maturity rating -> ordinal, missing at the median
    movies["maturity"] = movies["rating"].map(RATING_MAP)
    movies["rating_missing"] = movies["maturity"].isna().astype(int)
    maturity_median = movies["maturity"].median()
    movies["maturity"] = movies["maturity"].fillna(maturity_median)

    # Genres -> multi-hot, rare genres dropped
    movies["genre_list"] = movies["Genre"].apply(split_list)
    all_genres = sorted({g for gl in movies["genre_list"] for g in gl})
    genre_df = pd.DataFrame({f"g_{g}": movies["genre_list"].apply(lambda gl, g=g: int(g in gl))
                             for g in all_genres})
    genre_df = genre_df.loc[:, genre_df.sum() >= MIN_GENRE_COUNT]
    genre_cols = list(genre_df.columns)

    # Awards won by the title itself
    for c in AWARD_COLS:
        movies[c + "_n"] = movies[c].apply(lambda s: len(split_list(s)))
    movies["title_awards"] = movies[[c + "_n" for c in AWARD_COLS]].sum(axis=1)

    # Star power: awards held by cast and directors
    people = people.copy()
    for c in AWARD_COLS:
        people[c + "_n"] = people[c].apply(lambda s: len(split_list(s)))
    people["person_awards"] = people[[c + "_n" for c in AWARD_COLS]].sum(axis=1)
    person_awards = people.set_index("Person_ID")["person_awards"].to_dict()

    def talent_awards(ids):
        return sum(person_awards.get(i, 0) for i in split_list(ids))

    movies["actor_awards"] = movies["Actors_IDs"].apply(talent_awards)
    movies["director_awards"] = movies["Directors_IDs"].apply(talent_awards)

    movies["movie_age"] = CURRENT_YEAR - movies["year"]
    movies["n_platforms"] = movies["Platforms"].apply(lambda s: len(split_list(s)))

    for c in ["title_awards", "actor_awards", "director_awards", "n_platforms"]:
        movies[c + "_log"] = np.log1p(movies[c])

    X = pd.concat([movies[NUMERIC_FEATURES].reset_index(drop=True), genre_df], axis=1)
    params = {"rotten_slope": float(slope), "rotten_intercept": float(intercept),
              "maturity_median": float(maturity_median)}
    return movies, X, genre_cols, params


def scale_features(X, scaler, genre_cols):
    """StandardScaler on numerics, genres kept 0/1 and weighted."""
    Xs = X.copy()
    Xs[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    Xs[genre_cols] = X[genre_cols].values * GENRE_WEIGHT
    return Xs


def scenario_to_features(imdb_score, rotten_score, year, rating, genres,
                         title_awards, actor_awards, director_awards, n_platforms,
                         genre_cols):
    """Turn What-If inputs for one movie into a single unscaled feature row."""
    row = {
        "imdb_score": imdb_score,
        "rotten_score": rotten_score,
        "maturity": RATING_MAP[rating],
        "movie_age": CURRENT_YEAR - year,
        "title_awards_log": np.log1p(title_awards),
        "actor_awards_log": np.log1p(actor_awards),
        "director_awards_log": np.log1p(director_awards),
        "n_platforms_log": np.log1p(n_platforms),
    }
    for g in genre_cols:
        row[g] = int(g.replace("g_", "") in genres)
    return pd.DataFrame([row])[NUMERIC_FEATURES + genre_cols]
