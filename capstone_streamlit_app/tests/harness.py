"""
Loads the same data, models, facts and tools as app.py, without Streamlit,
so the assistant can be tested from the command line.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import llm_assistant as ai  # noqa: E402
from pipeline import (AWARD_CALIBER_PERSONA, MODELS_DIR, PERSONA_POLICY,  # noqa: E402
                      VIEWER_PERSONAS, build_features, load_raw, scale_features,
                      scenario_to_features)

PERSONA_ORDER = [VIEWER_PERSONAS[i] for i in range(len(VIEWER_PERSONAS))]

# Same default scenario as the app's "Oscar-winning epic" preset
DEFAULT_SCENARIO = dict(imdb=8.4, rotten=94, year=2003, rating="PG-13/TV-14",
                        genres=["Drama", "Adventure"], title_aw=6, actor_aw=4,
                        director_aw=3, platforms=2)


class Harness:
    def __init__(self, model_name="Neural Network (MLP)", threshold=0.5):
        raw, people = load_raw()
        self.raw_movies = raw[raw["type"] == "Movie"].copy()
        self.movies, X, self.genre_cols, self.params = build_features(raw, people)
        self.meta = joblib.load(MODELS_DIR / "pipeline_metadata.joblib")
        self.scaler = joblib.load(MODELS_DIR / "standard_scaler.joblib")
        kmeans = joblib.load(MODELS_DIR / "kmeans_personas.joblib")
        self.classifiers = {n: joblib.load(MODELS_DIR / f)
                            for n, f in self.meta["model_files"].items()}
        self.metrics = json.load(open(MODELS_DIR / "benchmark_metrics.json"))
        self.importance = json.load(open(MODELS_DIR / "feature_importance.json"))
        self.X = X
        self.X_scaled = scale_features(X, self.scaler, self.genre_cols)
        self.movies["cluster"] = kmeans.predict(self.X_scaled)
        self.movies["viewer_profile"] = self.movies["cluster"].map(VIEWER_PERSONAS)
        self.perf_df = pd.DataFrame(self.metrics["classifiers"])
        self.model_name, self.threshold = model_name, threshold
        self.model = self.classifiers[model_name]
        self.tools = ai.CatalogTools(
            self.movies, self.X_scaled, self.genre_cols, self.model, model_name, self.scaler,
            threshold, PERSONA_ORDER, AWARD_CALIBER_PERSONA, PERSONA_POLICY, scenario_to_features,
            scale_features, (self.params["rotten_slope"], self.params["rotten_intercept"]))

    def scenario(self, **s):
        s = {**DEFAULT_SCENARIO, **s}
        row = scenario_to_features(s["imdb"], s["rotten"], s["year"], s["rating"], s["genres"],
                                   s["title_aw"], s["actor_aw"], s["director_aw"], s["platforms"],
                                   self.genre_cols)
        scaled = scale_features(row, self.scaler, self.genre_cols)
        probs = self.model.predict_proba(scaled)[0]
        award = float(probs[PERSONA_ORDER.index(AWARD_CALIBER_PERSONA)])
        return {**s, "model": self.model_name, "threshold": self.threshold, "probs": probs,
                "persona": PERSONA_ORDER[int(np.argmax(probs))], "award_prob": award,
                "scaled_row": scaled.iloc[0].to_dict(),
                "out_of_range": ai.out_of_range(
                    {"imdb_score": s["imdb"], "rotten_score": s["rotten"], "year": s["year"],
                     "title_awards": s["title_aw"], "director_awards": s["director_aw"],
                     "actor_awards": s["actor_aw"], "platforms": s["platforms"]},
                    ai.training_ranges(self.movies))}

    def award_prob(self, current, **overrides):
        """Same as the app's award_prob_for_scenario: re-run the model with some inputs changed."""
        v = {**current, **overrides}
        row = scenario_to_features(v["imdb"], v["rotten"], v["year"], v["rating"], v["genres"],
                                   v["title_aw"], v["actor_aw"], v["director_aw"], v["platforms"],
                                   self.genre_cols)
        probs = self.model.predict_proba(scale_features(row, self.scaler, self.genre_cols))[0]
        return float(probs[PERSONA_ORDER.index(AWARD_CALIBER_PERSONA)])

    def facts(self, question="", **scenario):
        with_genres = self.movies.join(self.X[self.genre_cols])
        static = ai.build_static_context(
            with_genres, len(self.raw_movies), self.perf_df, self.metrics["kmeans"],
            self.importance, PERSONA_ORDER, self.genre_cols, PERSONA_POLICY, AWARD_CALIBER_PERSONA,
            raw_movie_rows=self.raw_movies, params=self.params, classifiers=self.classifiers)
        current = self.scenario(**scenario)
        live = ai.scenario_context(current, self.importance["award_caliber_logit_coef"],
                                   PERSONA_ORDER, lambda **o: self.award_prob(current, **o))
        return "\n\n".join([static, live])
