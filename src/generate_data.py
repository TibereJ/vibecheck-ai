"""
Generate synthetic Spotify-like music dataset for VibeCheck AI.
Features mirror real Spotify audio features API.
"""

import numpy as np
import pandas as pd

np.random.seed(42)


GENRES = [
    "techno", "house", "drum-and-bass", "ambient", "trance",
    "deep-house", "minimal", "melodic-techno", "afro-house", "hard-techno"
]

VIBE_PROFILES = {
    "euphoric": dict(energy=(0.75, 0.95), valence=(0.65, 0.95), tempo=(128, 145), danceability=(0.75, 0.95)),
    "dark":     dict(energy=(0.55, 0.85), valence=(0.05, 0.30), tempo=(130, 150), danceability=(0.60, 0.85)),
    "chill":    dict(energy=(0.15, 0.45), valence=(0.40, 0.75), tempo=(90,  120), danceability=(0.45, 0.70)),
    "hype":     dict(energy=(0.85, 1.00), valence=(0.45, 0.80), tempo=(140, 175), danceability=(0.80, 1.00)),
    "melancholic": dict(energy=(0.20, 0.55), valence=(0.05, 0.35), tempo=(85, 120), danceability=(0.35, 0.60)),
}

ADJECTIVES = ["Deep", "Dark", "Electric", "Neon", "Raw", "Cosmic", "Silent", "Burning", "Lost", "Rising"]
NOUNS = ["Night", "Signal", "Pulse", "Dream", "Echo", "Wave", "Void", "Ritual", "Current", "Horizon"]


def _sample(low, high, n):
    return np.random.uniform(low, high, n)


def generate_dataset(n_samples: int = 5000) -> pd.DataFrame:
    vibes = list(VIBE_PROFILES.keys())
    vibe_labels = np.random.choice(vibes, n_samples)

    rows = []
    for vibe in vibe_labels:
        p = VIBE_PROFILES[vibe]
        row = {
            "energy":        float(_sample(*p["energy"], 1)),
            "valence":       float(_sample(*p["valence"], 1)),
            "tempo":         float(_sample(*p["tempo"], 1)),
            "danceability":  float(_sample(*p["danceability"], 1)),
            "acousticness":  float(_sample(0.0, 0.3, 1)),
            "instrumentalness": float(_sample(0.5, 1.0, 1)),
            "liveness":      float(_sample(0.05, 0.4, 1)),
            "loudness":      float(_sample(-14, -3, 1)),
            "speechiness":   float(_sample(0.02, 0.12, 1)),
            "key":           int(np.random.randint(0, 12)),
            "mode":          int(np.random.randint(0, 2)),
            "duration_ms":   int(np.random.randint(180000, 480000)),
            "vibe":          vibe,
            "genre":         str(np.random.choice(GENRES)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Generate track names using adjective + noun combos
    adjs = np.random.choice(ADJECTIVES, n_samples)
    nouns = np.random.choice(NOUNS, n_samples)
    numbers = np.random.randint(1, 999, n_samples)
    df["track_name"] = [f"{a} {n} {num:03d}" for a, n, num in zip(adjs, nouns, numbers)]

    # Popularity score loosely correlated with energy + danceability
    df["popularity"] = np.clip(
        (df["energy"] * 40 + df["danceability"] * 40 + np.random.normal(10, 8, n_samples)),
        0, 100
    ).astype(int)

    return df


if __name__ == "__main__":
    df = generate_dataset(5000)
    df.to_csv("data/spotify_tracks.csv", index=False)
    print(f"✅ Dataset generated: {df.shape[0]} tracks")
    print(df.head())
