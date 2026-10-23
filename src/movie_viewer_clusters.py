"""
Data Mining Final Project - Movie Viewer Profile Clustering
-----------------------------------------------------------
Goal: cluster HBO Max MOVIES into groups that act as viewer-profile
classifiers, then describe what kind of viewer fits each group.

Pipeline
  1. Load & clean both datasets (movies only)
  2. Feature engineering (genres, maturity, quality, recency, awards, star power)
  3. Scale features
  4. Pick k (elbow + silhouette)
  5. K-Means clustering (+ PCA plot)
  6. Profile each cluster -> viewer persona
  7. Classifier: assign any movie / new viewer preferences to a profile
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
DATA_DIR = Path.home() / "Desktop" / "ing" / "sem 9" / "data mining" / "Capstone Project"
MOVIES_PATH = DATA_DIR / "HBO_MAX_Content_extended.csv"
PEOPLE_PATH = DATA_DIR / "People_Dataset.csv"

# ------------------------------------------------------------------
# 1. LOAD & CLEAN
# ------------------------------------------------------------------
movies = pd.read_csv(MOVIES_PATH)
people = pd.read_csv(PEOPLE_PATH)

movies = movies[movies["type"] == "Movie"].copy()
movies = movies.dropna(subset=["Genre", "imdb_score"]).reset_index(drop=True)
print(f"Movies after cleaning: {len(movies)}")

# Rotten score has many gaps -> impute from IMDb score (they correlate),
# and keep a flag so the model knows it was missing.
movies["rotten_missing"] = movies["rotten_score"].isna().astype(int)
known = movies.dropna(subset=["rotten_score"])
slope, intercept = np.polyfit(known["imdb_score"], known["rotten_score"], 1)
movies["rotten_score"] = movies["rotten_score"].fillna(
    (movies["imdb_score"] * slope + intercept).clip(0, 100))

# Maturity rating -> ordinal (missing = "Unrated", placed at median)
rating_map = {"G/TV-G": 0, "PG/TV-PG": 1, "PG-13/TV-14": 2, "R/TV-MA": 3}
movies["maturity"] = movies["rating"].map(rating_map)
movies["rating_missing"] = movies["maturity"].isna().astype(int)
movies["maturity"] = movies["maturity"].fillna(movies["maturity"].median())

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
def split_list(s, sep=","):
    return [x.strip() for x in str(s).split(sep) if x.strip()] if pd.notna(s) else []

# 2a. Genres -> multi-hot
movies["genre_list"] = movies["Genre"].apply(split_list)
mlb = MultiLabelBinarizer()
genre_df = pd.DataFrame(mlb.fit_transform(movies["genre_list"]),
                        columns=["g_" + g for g in mlb.classes_])
# drop very rare genres (noise for clustering)
genre_df = genre_df.loc[:, genre_df.sum() >= 15]

# 2b. Awards the movie itself won
award_cols = ["Academy_Awards", "Golden_Globe_Awards", "BAFTA_Awards", "Emmy_Awards"]
for c in award_cols:
    movies[c + "_n"] = movies[c].apply(lambda s: len(split_list(s)))
movies["title_awards"] = movies[[c + "_n" for c in award_cols]].sum(axis=1)

# 2c. Star power from People dataset (awards held by cast & directors)
for c in award_cols:
    people[c + "_n"] = people[c].apply(lambda s: len(split_list(s)))
people["person_awards"] = people[[c + "_n" for c in award_cols]].sum(axis=1)
person_awards = people.set_index("Person_ID")["person_awards"].to_dict()

def talent_awards(ids):
    return sum(person_awards.get(i, 0) for i in split_list(ids))

movies["actor_awards"] = movies["Actors_IDs"].apply(talent_awards)
movies["director_awards"] = movies["Directors_IDs"].apply(talent_awards)

# 2d. Recency and availability
movies["movie_age"] = 2026 - movies["year"]
movies["n_platforms"] = movies["Platforms"].apply(lambda s: len(split_list(s)))

# log-transform skewed counts
for c in ["title_awards", "actor_awards", "director_awards", "n_platforms"]:
    movies[c + "_log"] = np.log1p(movies[c])

numeric_features = ["imdb_score", "rotten_score", "maturity", "movie_age",
                    "title_awards_log", "actor_awards_log",
                    "director_awards_log", "n_platforms_log"]

X = pd.concat([movies[numeric_features].reset_index(drop=True), genre_df], axis=1)

# Genres are 0/1; scale only numerics, then down-weight genres slightly so
# 20+ genre columns don't drown out quality/maturity/era.
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
GENRE_WEIGHT = 1.5
X_scaled[genre_df.columns] = genre_df.values * GENRE_WEIGHT
print(f"Feature matrix: {X_scaled.shape}")

# ------------------------------------------------------------------
# 3. CHOOSE K
# ------------------------------------------------------------------
ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.0f}  silhouette={sils[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertias, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k")
ax[1].plot(ks, sils, "o-"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("choose_k.png", dpi=120); plt.close()

# Silhouette alone favors tiny k; for interpretable personas aim for 4-7.
# Change K after looking at choose_k.png.
K = 6

# ------------------------------------------------------------------
# 4. FINAL CLUSTERING
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=K, n_init=50, random_state=RANDOM_STATE)
movies["cluster"] = kmeans.fit_predict(X_scaled)
print(f"\nFinal silhouette (k={K}): {silhouette_score(X_scaled, movies['cluster']):.3f}")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)
plt.figure(figsize=(9, 7))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=movies["cluster"], cmap="tab10", s=12, alpha=0.7)
plt.legend(*sc.legend_elements(), title="Cluster")
plt.title(f"Movie clusters (PCA, {pca.explained_variance_ratio_.sum():.0%} variance shown)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig("clusters_pca.png", dpi=120); plt.close()

# ------------------------------------------------------------------
# 5. PROFILE EACH CLUSTER
# ------------------------------------------------------------------
inv_rating = {v: k for k, v in rating_map.items()}
profile_cols = ["imdb_score", "rotten_score", "maturity", "year",
                "title_awards", "actor_awards", "director_awards", "n_platforms"]
profile = movies.groupby("cluster")[profile_cols].mean().round(2)
profile["size"] = movies["cluster"].value_counts().sort_index()

genre_share = pd.concat([movies[["cluster"]], genre_df], axis=1).groupby("cluster").mean()
overall_genre = genre_df.mean()

def top_genres(c, n=4):
    # "lift" = how over-represented a genre is vs. the whole catalog
    common = genre_share.loc[c][genre_share.loc[c] >= 0.2].index  # in >=20% of cluster
    lift = (genre_share.loc[c, common] / overall_genre[common]).sort_values(ascending=False)
    return [f"{g.replace('g_', '')} ({genre_share.loc[c, g]:.0%})" for g in lift.index[:n]]

def describe(c):
    r = profile.loc[c]
    traits = []
    q = movies["imdb_score"].quantile([.33, .66])
    traits.append("acclaimed" if r.imdb_score >= q.iloc[1] else
                  "casual / crowd-pleaser" if r.imdb_score <= q.iloc[0] else "mid-quality")
    traits.append("family-friendly" if r.maturity < 1.5 else "mature" if r.maturity > 2.4 else "teen/general")
    traits.append("classic" if r.year < 1995 else "modern" if r.year > 2012 else "2000s-era")
    if r.title_awards > movies["title_awards"].mean() * 2:
        traits.append("award-winning")
    if r.actor_awards + r.director_awards > (movies["actor_awards"] + movies["director_awards"]).mean() * 1.5:
        traits.append("star-driven")
    return traits

print("\n================ CLUSTER PROFILES ================")
summary_rows = []
for c in range(K):
    r = profile.loc[c]
    g = top_genres(c)
    t = describe(c)
    examples = (movies[movies.cluster == c]
                .sort_values("imdb_score", ascending=False)["title"].head(5).tolist())
    print(f"\nCluster {c}  (n={int(r['size'])})")
    print(f"  Genres  : {', '.join(g)}")
    print(f"  Traits  : {', '.join(t)}")
    print(f"  IMDb {r.imdb_score} | RT {r.rotten_score} | avg year {int(r.year)} | "
          f"rating ~{inv_rating[round(r.maturity)]}")
    print(f"  Examples: {examples}")
    summary_rows.append({"cluster": c, "size": int(r["size"]), "top_genres": ", ".join(g),
                         "traits": ", ".join(t), "examples": " | ".join(examples)})

summary = pd.DataFrame(summary_rows).merge(profile.reset_index(), on=["cluster", "size"])
summary.to_csv("cluster_profiles.csv", index=False)

# Fill these in AFTER reading the printed profiles - this is your
# analysis/conclusion step, so name them based on what you see.
VIEWER_PERSONAS = {c: f"Persona {c}" for c in range(K)}
# e.g. VIEWER_PERSONAS = {0: "Family Movie Night", 1: "Prestige Cinephile", ...}
movies["viewer_profile"] = movies["cluster"].map(VIEWER_PERSONAS)

# Genre heatmap per cluster
plt.figure(figsize=(14, 4))
plt.imshow(genre_share.values, aspect="auto", cmap="Blues")
plt.yticks(range(K), [f"C{c}" for c in range(K)])
plt.xticks(range(len(genre_share.columns)),
           [g.replace("g_", "") for g in genre_share.columns], rotation=60, ha="right")
plt.colorbar(label="Share of movies in cluster")
plt.title("Genre composition by cluster")
plt.tight_layout(); plt.savefig("genre_heatmap.png", dpi=120); plt.close()

movies[["title", "year", "Genre", "rating", "imdb_score", "cluster", "viewer_profile"]] \
    .to_csv("movies_with_clusters.csv", index=False)

# ------------------------------------------------------------------
# 6. CLASSIFIER: place a new viewer into a profile
# ------------------------------------------------------------------
def classify_viewer(liked_titles):
    """Given movies a viewer liked, return the profile they fit best."""
    liked = movies[movies["title"].str.lower().isin([t.lower() for t in liked_titles])]
    if liked.empty:
        return "No matching titles found"
    counts = liked["cluster"].value_counts(normalize=True)
    best = counts.idxmax()
    return f"{VIEWER_PERSONAS[best]} (cluster {best}, {counts.max():.0%} of liked titles)"

def recommend(cluster_id, n=10, exclude=()):
    pool = movies[(movies.cluster == cluster_id) & (~movies.title.isin(exclude))]
    return pool.sort_values("imdb_score", ascending=False)["title"].head(n).tolist()

if __name__ == "__main__":
    demo = movies["title"].sample(3, random_state=1).tolist()
    print(f"\nDemo viewer liked: {demo}")
    print("Profile:", classify_viewer(demo))
    print("\nSaved: choose_k.png, clusters_pca.png, genre_heatmap.png, "
          "cluster_profiles.csv, movies_with_clusters.csv")
