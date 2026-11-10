"""
PrestigePick - offline training script (run once before launching the app)
--------------------------------------------------------------------------
    python train_models.py

1. Builds the movie feature matrix (pipeline.py)
2. Fits the StandardScaler and K-Means (k=6) -> viewer persona labels
3. Trains three persona classifiers on an 80/20 stratified split:
   Decision Tree, Logistic Regression and an Artificial Neural Network (MLP)
4. Saves every fitted object to models/*.joblib plus the benchmark metrics

The app (app.py) only LOADS these files; it never retrains.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline import (K, MODELS_DIR, NUMERIC_FEATURES, RANDOM_STATE,
                      VIEWER_PERSONAS, build_features, load_raw, scale_features)

MODELS_DIR.mkdir(exist_ok=True)

# 1. Features ---------------------------------------------------------------
movies, X, genre_cols, params = build_features(*load_raw())
print(f"Movies: {len(movies)} | features: {X.shape[1]}")

# 2. Scaler + K-Means personas ---------------------------------------------
scaler = StandardScaler().fit(X[NUMERIC_FEATURES])
X_scaled = scale_features(X, scaler, genre_cols)

kmeans = KMeans(n_clusters=K, n_init=50, random_state=RANDOM_STATE).fit(X_scaled)
y = pd.Series(kmeans.labels_, name="cluster")
sil = silhouette_score(X_scaled, y)
aris = [adjusted_rand_score(y, KMeans(n_clusters=K, n_init=20, random_state=s).fit_predict(X_scaled))
        for s in range(10)]
print(f"K-Means k={K}: silhouette={sil:.3f}, stability mean ARI={np.mean(aris):.3f}")
print(y.map(VIEWER_PERSONAS).value_counts().to_string())

# 3. Persona classifiers ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

models = {
    "Decision Tree (CART)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                                   random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                          learning_rate_init=1e-3, max_iter=600,
                                          early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=25, random_state=RANDOM_STATE),
}
files = {
    "Decision Tree (CART)": "decision_tree_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
    "Neural Network (MLP)": "ann_mlp_model.joblib",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    cv_f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")
    rows.append({
        "Model": name,
        "Family": "Neural Network" if "Neural" in name else "Classical",
        "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
        "Test Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
        "ROC-AUC (OvR)": roc_auc_score(y_test, prob, multi_class="ovr"),
        "5-fold CV F1": cv_f1.mean(),
    })
    joblib.dump(model, MODELS_DIR / files[name])
    print(f"{name}: test acc={rows[-1]['Test Accuracy']:.3f}  F1={rows[-1]['F1 (macro)']:.3f}  "
          f"AUC={rows[-1]['ROC-AUC (OvR)']:.3f}  CV F1={cv_f1.mean():.3f}")

# 4. Save artifacts ---------------------------------------------------------
joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")
joblib.dump(kmeans, MODELS_DIR / "kmeans_personas.joblib")
joblib.dump({
    "genre_cols": genre_cols,
    "feature_cols": list(X.columns),
    "model_files": files,
    "test_index": list(X_test.index),
    "params": params,
}, MODELS_DIR / "pipeline_metadata.joblib")

with open(MODELS_DIR / "benchmark_metrics.json", "w") as f:
    json.dump({"classifiers": rows,
               "kmeans": {"k": K, "silhouette": sil, "inertia": float(kmeans.inertia_),
                          "stability_mean_ari": float(np.mean(aris)),
                          "stability_min_ari": float(np.min(aris))}}, f, indent=2)

print(f"\nSaved models to {MODELS_DIR}")
