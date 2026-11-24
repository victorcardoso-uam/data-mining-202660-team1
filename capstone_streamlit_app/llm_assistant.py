"""
PrestigePick AI Assistant - LLM chatbot grounded in the app's own data and models
---------------------------------------------------------------------------------
The LLM never sees the raw CSV and never invents numbers: on every question the
app builds a fact sheet (dataset profile, persona profiles, benchmark metrics,
feature importance, the live What-If decision and any catalog movie named in the
question) and the model answers ONLY from it.

Providers (set in .streamlit/secrets.toml or environment variables):
  LLM_PROVIDER = "anthropic"  -> Claude API (needs ANTHROPIC_API_KEY)
  LLM_PROVIDER = "openai"     -> any OpenAI-compatible API (OPENAI_API_KEY,
                                 OPENAI_BASE_URL, LLM_MODEL): OpenAI, Google
                                 Gemini, Groq, OpenRouter or a local Ollama.
"""

import os
import re

import numpy as np
import pandas as pd
import requests

FEATURE_LABELS = {
    "imdb_score": "IMDb score",
    "rotten_score": "Rotten Tomatoes score",
    "maturity": "Maturity rating (0=G ... 3=R)",
    "movie_age": "Movie age (2026 - year)",
    "title_awards_log": "Awards won by the title",
    "actor_awards_log": "Awards held by the cast",
    "director_awards_log": "Awards held by the director(s)",
    "n_platforms_log": "Number of streaming platforms",
}


def label(col):
    return FEATURE_LABELS.get(col, f"Genre: {col[2:]}" if col.startswith("g_") else col)


# -----------------------------------------------------------------------------
# Provider configuration
# -----------------------------------------------------------------------------
def get_config(secrets):
    """Read provider settings from st.secrets first, then environment variables."""
    def get(key, default=None):
        try:
            if key in secrets:
                return secrets[key]
        except Exception:  # no secrets.toml at all
            pass
        return os.environ.get(key, default)

    provider = (get("LLM_PROVIDER") or ("anthropic" if get("ANTHROPIC_API_KEY") else
                                        "openai" if get("OPENAI_API_KEY") or get("OPENAI_BASE_URL")
                                        else "")).lower()
    if provider == "anthropic":
        return {"provider": "anthropic", "api_key": get("ANTHROPIC_API_KEY"),
                "model": get("LLM_MODEL", "claude-haiku-5-5"),
                "base_url": get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")}
    if provider == "openai":
        return {"provider": "openai", "api_key": get("OPENAI_API_KEY", ""),
                "model": get("LLM_MODEL", "gpt-4o-mini"),
                "base_url": get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")}
    return None


def is_ready(cfg):
    if not cfg:
        return False
    if cfg["provider"] == "anthropic":
        return bool(cfg["api_key"])
    # OpenAI-compatible: a key is required except for local servers such as Ollama
    return bool(cfg["api_key"]) or "localhost" in cfg["base_url"] or "127.0.0.1" in cfg["base_url"]


def ask_llm(cfg, system, messages, max_tokens=700, timeout=60):
    """Send the chat to the configured provider and return the answer text."""
    if cfg["provider"] == "anthropic":
        r = requests.post(
            f"{cfg['base_url'].rstrip('/')}/v1/messages",
            headers={"x-api-key": cfg["api_key"], "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": cfg["model"], "max_tokens": max_tokens, "temperature": 0.2,
                  "system": system, "messages": messages},
            timeout=timeout)
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")

    headers = {"content-type": "application/json"}
    if cfg["api_key"]:
        headers["authorization"] = f"Bearer {cfg['api_key']}"
    r = requests.post(
        f"{cfg['base_url']}/chat/completions", headers=headers,
        json={"model": cfg["model"], "max_tokens": max_tokens, "temperature": 0.2,
              "messages": [{"role": "system", "content": system}] + messages},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------------
# Grounding: the fact sheet the LLM answers from
# -----------------------------------------------------------------------------
def _table(df):
    return df.to_csv(index=False).strip()


def persona_profiles(movies, persona_order, genre_cols):
    rows = []
    for p in persona_order:
        g = movies[movies["viewer_profile"] == p]
        genre_share = g[genre_cols].mean().sort_values(ascending=False) if genre_cols else pd.Series()
        top_genres = ", ".join(f"{c[2:]} {v:.0%}" for c, v in genre_share.head(3).items())
        rows.append({
            "persona": p, "movies": len(g), "share": f"{len(g) / len(movies):.1%}",
            "mean_imdb": round(g["imdb_score"].mean(), 2),
            "mean_rotten": round(g["rotten_score"].mean(), 1),
            "median_year": int(g["year"].median()),
            "pct_R_rated": f"{(g['rating'] == 'R/TV-MA').mean():.0%}",
            "mean_title_awards": round(g["title_awards"].mean(), 2),
            "pct_won_award": f"{(g['title_awards'] > 0).mean():.0%}",
            "mean_director_awards": round(g["director_awards"].mean(), 2),
            "top_genres": top_genres,
            "top_rated_examples": " | ".join(g.sort_values("imdb_score", ascending=False)["title"].head(3)),
        })
    return pd.DataFrame(rows)


PROFILE_COLS = ["imdb_score", "rotten_score", "maturity", "year", "title_awards",
                "actor_awards", "director_awards", "n_platforms"]


def eda_context(raw_movie_rows, movies, params, genre_cols):
    """Capability 1: dataset dimensions, missingness, distributions, correlations."""
    miss = raw_movie_rows[["rating", "imdb_score", "rotten_score", "Genre",
                           "Actors_IDs", "Directors_IDs"]].isna().sum()
    desc = movies[PROFILE_COLS].describe().T[["mean", "50%", "std", "min", "max"]].round(2)
    desc["skew"] = movies[PROFILE_COLS].skew().round(2)
    desc = desc.rename(columns={"50%": "median"}).reset_index().rename(columns={"index": "feature"})
    corr = movies[PROFILE_COLS].corr()
    pairs = sorted(((a, b, corr.loc[a, b]) for i, a in enumerate(PROFILE_COLS)
                    for b in PROFILE_COLS[i + 1:]), key=lambda t: -abs(t[2]))[:6]
    genre_share = movies[genre_cols].mean().sort_values(ascending=False) if genre_cols else pd.Series()
    return "\n".join([
        "## Dataset dimensions",
        f"Raw HBO Max file: {len(raw_movie_rows)} movies (TV shows excluded). After cleaning: "
        f"{len(movies)} movies x 30 model features (8 numeric + {len(genre_cols)} multi-hot genres). "
        "Second source: People dataset (cast and directors with their awards), joined by person ID.",
        "",
        "## Missing values in the raw movie rows and their treatment",
        f"rating missing {miss['rating']} ({miss['rating'] / len(raw_movie_rows):.0%}) -> ordinal median "
        "fill + rating_missing flag; "
        f"rotten_score missing {miss['rotten_score']} ({miss['rotten_score'] / len(raw_movie_rows):.0%}) -> "
        f"estimated from IMDb with a linear fit (RT = {params['rotten_slope']:.1f} x IMDb "
        f"{params['rotten_intercept']:+.1f}, clipped 0-100) + rotten_missing flag; "
        f"imdb_score missing {miss['imdb_score']} and Genre missing {miss['Genre']} -> rows dropped "
        f"({len(raw_movie_rows) - len(movies)} dropped in total); "
        f"Actors_IDs missing {miss['Actors_IDs']}, Directors_IDs missing {miss['Directors_IDs']} -> "
        "counted as 0 awards. Award columns are empty when a movie won nothing (counted as 0).",
        "",
        "## Distributions of the main variables (after cleaning) (CSV)", _table(desc),
        "Skew > 1 means a long right tail: most movies have 0 awards, a few have many, which is why "
        "award counts were log-transformed before scaling. All numeric features were then standardized "
        "(StandardScaler: mean 0, std 1).",
        "Most common genres: " + ", ".join(f"{c[2:]} {v:.0%}" for c, v in genre_share.head(6).items()) + ".",
        "",
        "## Strongest Pearson correlations between variables",
        "; ".join(f"{a} vs {b}: {r:+.2f}" for a, b, r in pairs)
        + ". Note: the IMDb-Rotten Tomatoes correlation is partly inflated because 27% of Rotten "
          "scores were imputed from IMDb.",
    ])


def architecture_context(classifiers, perf_df):
    """Capability 2: how each model is built and how to read the comparison."""
    lines = ["## Model architectures (all trained on the same stratified 80% split, 1,257 movies)"]
    for name, m in classifiers.items():
        if hasattr(m, "get_depth"):
            lines.append(f"- {name}: CART decision tree, max_depth 6, min_samples_leaf 5; final tree has "
                         f"depth {m.get_depth()} and {m.get_n_leaves()} leaves. Splits one feature at a time.")
        elif hasattr(m, "hidden_layer_sizes"):
            lines.append(f"- {name}: feed-forward neural network, 2 hidden layers {m.hidden_layer_sizes} "
                         f"with {m.activation} activation, softmax output over 6 personas, Adam optimizer, "
                         f"L2 penalty alpha {m.alpha}, early stopping on a 15% validation split. Trained "
                         f"{m.n_iter_} epochs; best validation accuracy {m.best_validation_score_:.3f}.")
        else:
            lines.append(f"- {name}: multinomial logistic regression (softmax over 6 personas), L2 "
                         f"regularization C={m.C}, solver {m.solver}. One linear score per persona.")
    best = perf_df.sort_values("F1 (macro)", ascending=False)["Model"].tolist()
    n_params = {}
    for name, m in classifiers.items():
        if hasattr(m, "coefs_"):
            n_params["ann"] = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
        elif hasattr(m, "coef_"):
            n_params["lr"] = m.coef_.size + m.intercept_.size
    lines += ["", "## Team's interpretation of the benchmark",
              f"Ranking by test macro F1: {' > '.join(best)}.",
              "- Logistic Regression wins because the personas were created by K-Means, which separates "
              "groups by straight-line (Euclidean) distance on the same standardized features. Boundaries "
              "between such groups are close to linear, which is exactly what logistic regression draws.",
              "- The 2-layer ANN can draw curved boundaries, but here that extra flexibility is not needed; "
              f"it is 2.3 F1 points behind with {n_params.get('ann', 0):,} parameters vs "
              f"{n_params.get('lr', 0):,} for logistic regression, and would need more data to use that "
              "capacity. Its ROC-AUC (0.998) is almost identical.",
              "- The Decision Tree is weakest because it splits on one feature at a time, so it struggles "
              "with personas defined by combinations of many genre flags; its main error is labeling 26 "
              "test documentaries as mature dramas.",
              "- No R-squared is reported: every model predicts a category (persona), not a number.",
              "- Recommendation: deploy Logistic Regression for accuracy and explainability, keep the ANN as "
              "a challenger model. Train-test accuracy gaps are small (Logistic Regression 0.997 vs 0.962, "
              "ANN 0.982 vs 0.940) and 5-fold CV agrees, so neither is overfitted."]
    return "\n".join(lines)


def governance_context(movies, kmeans_metrics, award_persona):
    """Capability 4: limitations, possible biases, validation protocol."""
    prestige = movies[movies["viewer_profile"] == award_persona]
    old_all, old_p = (movies["year"] < 1980).mean(), (prestige["year"] < 1980).mean()
    return "\n".join([
        "## Responsible AI: limitations",
        "- No real viewing or engagement data: high_engagement_flag is a proxy (membership in the "
        f"{award_persona} persona), not observed viewer behavior. It must be validated with an A/B test "
        "on real clicks before it drives the home page.",
        "- The classifiers learn personas that K-Means created from the same features, so their high "
        "scores show they reproduce the clustering, not that the clustering matches real viewers.",
        f"- K-Means silhouette is {kmeans_metrics['silhouette']:.3f} (weak separation): many movies sit "
        "between personas, so borderline titles deserve human review.",
        f"- Class imbalance: only {len(prestige)} of {len(movies)} movies ({len(prestige) / len(movies):.1%}) "
        "are Award Caliber; metrics are macro-averaged so the small class counts equally.",
        "- 27% of Rotten Tomatoes scores are imputed and 25% of ratings are filled with the median.",
        "- The catalog ends in 2020; new releases have had less time to win awards.",
        "",
        "## Responsible AI: potential feature biases",
        "- Awards (Oscar, Golden Globe, BAFTA, Emmy) are the strongest drivers. These bodies historically "
        "favor English-language, US/UK and older prestige films, so the carousel may under-represent "
        "international, independent and recent titles and documentaries.",
        f"- Era: {old_p:.0%} of Award Caliber titles are pre-1980 vs {old_all:.0%} of the catalog "
        "(movie age is the 3rd most important feature).",
        "- Star power counts the awards of the cast and directors, which rewards already-famous people "
        "(a popularity / Matthew-effect bias).",
        "",
        "## Responsible AI: validation protocol used",
        "- Stratified 80/20 train-test split (random_state 42); all reported metrics are on the 315-movie "
        "held-out test set, never the training data.",
        "- 5-fold stratified cross-validation on every classifier (F1 agrees with test F1 within 0.01).",
        f"- Cluster stability: K-Means re-run with 10 random seeds, mean ARI {kmeans_metrics['stability_mean_ari']:.3f} "
        f"(min {kmeans_metrics['stability_min_ari']:.3f}).",
        "- Overfitting checks: train vs test accuracy, ANN early stopping on a validation split.",
        "- Explainability: permutation importance on the test set and logistic regression coefficients.",
        "- The scaler and models are frozen .joblib files, so every prediction is reproducible.",
        "- Human oversight: the cutoff slider lets editors trade precision for recall; titles near the "
        "cutoff, or where K-Means and the classifier disagree, should be reviewed by a person.",
        "- This assistant answers only from these facts; it can be wrong and does not replace review.",
    ])


def build_static_context(movies, raw_count, perf_df, kmeans_metrics, importance,
                         persona_order, genre_cols, persona_policy, award_persona,
                         raw_movie_rows=None, params=None, classifiers=None):
    """Facts that do not change while the app runs (cached by the caller)."""
    lines = ["## Project",
             "PrestigePick, Team 1. Client: HBO Max UX & Content Team. Goal: auto-curate the "
             "'Award Caliber' home-page carousel. K-Means (k=6) groups movies into viewer personas; "
             "Decision Tree, Logistic Regression and an ANN (MLP 64-32) classify any movie into a "
             f"persona. high_engagement_flag = 1 when P({award_persona}) >= the cutoff.",
             "",
             "## Dataset",
             f"HBO Max movie catalog: {raw_count} movies, {len(movies)} kept after dropping rows with no "
             f"genre or IMDb score. Years {int(movies['year'].min())}-{int(movies['year'].max())}. "
             f"Mean IMDb {movies['imdb_score'].mean():.2f}, mean Rotten Tomatoes "
             f"{movies['rotten_score'].mean():.1f}. {(movies['title_awards'] > 0).mean():.1%} of movies "
             "won at least one major award (Oscar, Golden Globe, BAFTA, Emmy). Missing Rotten Tomatoes "
             "scores were estimated from IMDb with a linear fit; missing ratings filled with the median.",
             "",
             "## Persona profiles (CSV)", _table(persona_profiles(movies, persona_order, genre_cols)),
             "",
             "## Carousel policy per persona"]
    lines += [f"- {p}: {persona_policy[p]}" for p in persona_order]

    bench = perf_df[["Model", "Test Accuracy", "Precision (macro)", "Recall (macro)",
                     "F1 (macro)", "ROC-AUC (OvR)", "5-fold CV F1"]].round(3)
    lines += ["", "## Benchmark on the 20% held-out test set (315 movies) (CSV)", _table(bench),
              f"K-Means silhouette {kmeans_metrics['silhouette']:.3f}; stability across 10 seeds: "
              f"mean ARI {kmeans_metrics['stability_mean_ari']:.3f}.",
              "", "## Feature importance: permutation importance (drop in test macro F1 when the "
                  "feature is shuffled; higher = more important)"]
    for model, scores in importance["permutation_f1_drop"].items():
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:8]
        lines.append(f"- {model}: " + "; ".join(f"{label(c)} {v:.3f}" for c, v in top))
    coef = sorted(importance["award_caliber_logit_coef"].items(), key=lambda kv: -kv[1])
    lines += ["", f"## Logistic Regression coefficients for the {award_persona} persona "
                  "(standardized inputs; + pushes a title INTO the Award Caliber carousel, - keeps it out)",
              "Strongest positive: " + "; ".join(f"{label(c)} {v:+.2f}" for c, v in coef[:6]),
              "Strongest negative: " + "; ".join(f"{label(c)} {v:+.2f}" for c, v in coef[-6:][::-1])]
    if raw_movie_rows is not None:
        lines += ["", eda_context(raw_movie_rows, movies, params, genre_cols)]
    if classifiers is not None:
        lines += ["", architecture_context(classifiers, perf_df)]
    lines += ["", governance_context(movies, kmeans_metrics, award_persona)]
    return "\n".join(lines)


def flip_advice(s, predict_award):
    """Capability 3: which single slider change would flip the carousel decision.

    predict_award(**overrides) -> Award Caliber probability with those inputs changed.
    """
    inside = s["award_prob"] >= s["threshold"]
    checks = [("title_aw", "awards won by the title", range(0, 16)),
              ("director_aw", "awards held by the director(s)", range(0, 16)),
              ("rotten", "Rotten Tomatoes score", range(0, 101, 5)),
              ("imdb", "IMDb score", [round(x, 1) for x in np.arange(1.0, 10.01, 0.5)])]
    out = []
    for key, name, values in checks:
        cur = s[key]
        # Walk away from the current value, nearest first: down if inside, up if outside
        cands = sorted((v for v in values if v < cur), reverse=True) if inside \
            else sorted(v for v in values if v > cur)
        hit = next((v for v in cands if (predict_award(**{key: v}) >= s["threshold"]) != inside), None)
        if hit is None:
            out.append(f"changing {name} alone (currently {cur}) cannot flip the decision")
        else:
            out.append(f"{name} {'down' if inside else 'up'} from {cur} to {hit} flips it "
                       f"({predict_award(**{key: hit}):.0%})")
    verb = "drop OUT of" if inside else "move INTO"
    return f"Smallest single-slider change that would {verb} the carousel (others held fixed): " \
           + "; ".join(out) + "."


def scenario_context(s, coef, persona_order, predict_award=None):
    """The live What-If scenario from Tab 4 and why the model decided what it did."""
    probs = "; ".join(f"{p} {pr:.1%}" for p, pr in zip(persona_order, s["probs"]))
    lines = ["## Live What-If scenario (Tab 4, right now)",
             f"Inputs: IMDb {s['imdb']}, Rotten Tomatoes {s['rotten']}, release year {s['year']}, "
             f"rating {s['rating']}, genres {', '.join(s['genres']) or 'none'}, awards won by the title "
             f"{s['title_aw']}, director awards {s['director_aw']}, cast awards {s['actor_aw']}, "
             f"platforms {s['platforms']}.",
             f"Active model: {s['model']}. Cutoff: {s['threshold']:.2f}.",
             f"Persona probabilities: {probs}.",
             f"Predicted persona: {s['persona']}. Award Caliber probability {s['award_prob']:.1%} -> "
             f"{'GREEN: add to the Award Caliber carousel (high_engagement_flag = 1)' if s['award_prob'] >= s['threshold'] else 'RED: not Award Caliber (high_engagement_flag = 0)'}."]
    # Decision logic: Logistic Regression log-odds contributions = coefficient x scaled value
    contrib = {c: coef[c] * float(v) for c, v in s["scaled_row"].items() if c in coef}
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
    lines.append("Why (Logistic Regression log-odds contributions toward Award Caliber for this "
                 "scenario, coefficient x standardized value): "
                 + "; ".join(f"{label(c)} {v:+.2f}" for c, v in top))
    margin = s["award_prob"] - s["threshold"]
    lines.append(f"Distance from the cutoff: {margin:+.1%} "
                 f"({'borderline, send to human review' if abs(margin) < 0.15 else 'clear decision'}).")
    if predict_award is not None:
        lines.append(flip_advice(s, predict_award))
    return "\n".join(lines)


def movie_lookup(question, movies, predict_award_prob, limit=5):
    """Catalog movies whose title appears in the question, with the model's decision."""
    q = question.lower()
    titles = movies["title"].dropna()
    hits = movies[titles.str.len().ge(4) & titles.str.lower().apply(
        lambda t: re.search(r"\b" + re.escape(t) + r"\b", q) is not None)]
    if hits.empty:
        return ""
    hits = hits.assign(_len=hits["title"].str.len()).sort_values("_len", ascending=False).head(limit)
    rows = []
    for idx, m in hits.iterrows():
        rows.append(f"- {m['title']} ({int(m['year'])}, {m['rating']}): IMDb {m['imdb_score']}, "
                    f"RT {m['rotten_score']:.0f}, title awards {int(m['title_awards'])}, director awards "
                    f"{int(m['director_awards'])}, genres {m['Genre']}. K-Means persona: {m['viewer_profile']}. "
                    f"Active model's Award Caliber probability: {predict_award_prob(idx):.1%}.")
    return "## Catalog movies named in the question\n" + "\n".join(rows)


SYSTEM_PROMPT = """You are the PrestigePick AI Assistant inside a Streamlit dashboard built by \
Team 1 for the HBO Max UX & Content Team. Your users are non-technical content managers and an \
executive jury.

You cover four areas:
1. Dataset & EDA: dimensions, missing-value treatment, distributions, correlations, personas.
2. Models & metrics: architectures, why one model beats another on held-out F1, feature importance.
3. Real-time scenario advice: for the live What-If scenario, give plain-English operational guidance: \
the decision, why, how close it is to the cutoff, which slider change would flip it, and the \
recommended carousel row.
4. Responsible AI & governance: limitations, potential biases, validation protocol, when a human \
should review.

Rules:
- Use ONLY the facts below. Quote numbers exactly as given. Never invent a movie, number or metric.
- If the facts do not contain the answer, say so in one sentence and point to the dashboard tab \
that would show it.
- For audit questions, be candid about weaknesses; do not oversell the models.
- Be concise: at most about 170 words, short paragraphs or bullets, plain language. No code unless asked.

FACTS
-----
{facts}
"""


def suggested_questions():
    # One starter per required capability (+ a catalog lookup)
    return [
        "How was missing data handled?",
        "Why does Logistic Regression beat the ANN?",
        "What should I do with this What-If title?",
        "What biases could this model have?",
        "Is Casablanca Award Caliber?",
    ]
