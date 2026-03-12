"""
VibeCheck AI — ML Pipeline
- KMeans clustering for vibe detection
- Neural network for danceability prediction
- NLP sentiment on track names
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

AUDIO_FEATURES = [
    "energy", "valence", "tempo", "danceability",
    "acousticness", "instrumentalness", "liveness",
    "loudness", "speechiness"
]

VIBE_CLUSTER_NAMES = {
    0: "🔥 Hype",
    1: "🌙 Dark",
    2: "✨ Euphoric",
    3: "🌊 Chill",
    4: "💜 Melancholic",
}


def preprocess(df: pd.DataFrame):
    """Scale audio features."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES])
    return X, scaler


def train_clustering(X: np.ndarray, n_clusters: int = 5):
    """Train KMeans and return model + labels + silhouette score."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return kmeans, labels, score


def find_optimal_k(X: np.ndarray, k_range=range(2, 10)):
    """Elbow method + silhouette to find best k."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, lbl))
    return list(k_range), inertias, silhouettes


def train_neural_network(df: pd.DataFrame):
    """
    Train a Keras neural net to predict danceability from other audio features.
    Returns model, history, test metrics.
    """
    import tensorflow as tf
    from tensorflow import keras

    features = [f for f in AUDIO_FEATURES if f != "danceability"]
    X = df[features].values
    y = df["danceability"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=64,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, history, {"mae": mae, "r2": r2, "y_test": y_test, "y_pred": y_pred}


def analyze_sentiment_simple(track_names: pd.Series) -> pd.Series:
    """
    Lightweight NLP: keyword-based sentiment scoring on track names.
    (No heavy transformer needed for demo — uses curated lexicon)
    """
    positive_words = {
        "rise", "rising", "dream", "love", "light", "cosmic", "burning",
        "electric", "neon", "bright", "euphoric", "magic", "golden"
    }
    negative_words = {
        "dark", "lost", "void", "silent", "dead", "ghost", "shadow",
        "cold", "raw", "broken", "empty", "hollow"
    }

    def score(name: str) -> float:
        tokens = name.lower().split()
        pos = sum(1 for t in tokens if t in positive_words)
        neg = sum(1 for t in tokens if t in negative_words)
        total = pos + neg
        if total == 0:
            return 0.5
        return pos / total

    return track_names.apply(score)


def get_vibe_label(cluster_id: int) -> str:
    return VIBE_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
