# 🎧 VibeCheck AI — Music Mood & Energy Analyzer

> AI-powered tool for DJs and music analysts to classify tracks by vibe using Machine Learning and Deep Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🎯 What It Does

VibeCheck AI analyzes music audio features and automatically classifies tracks into **vibe clusters** — Euphoric, Dark, Chill, Hype, Melancholic — using unsupervised machine learning. It also predicts danceability with a deep neural network and performs NLP sentiment analysis on track names.

**Perfect for:**
- DJs building set playlists by mood
- Music analysts studying audio feature patterns
- Anyone curious about what makes a track "hype" vs "chill"

---

## 🧠 ML Pipeline

| Component | Method | Purpose |
|---|---|---|
| **Preprocessing** | StandardScaler | Normalize audio features |
| **Clustering** | KMeans (K=5) | Discover vibe groups |
| **Evaluation** | Silhouette Score + Elbow | Find optimal K |
| **Dimensionality reduction** | PCA (2D) | Visualize clusters |
| **Prediction** | Neural Network (Keras) | Predict danceability |
| **NLP** | Keyword sentiment scoring | Analyze track name tone |

### Neural Network Architecture

```
Input (8 features)
    → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    → Dense(64, ReLU)  → BatchNorm → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, Sigmoid)  →  Danceability score [0, 1]
```

---

## 📊 Audio Features Used

| Feature | Description |
|---|---|
| `energy` | Intensity and activity (0–1) |
| `valence` | Musical positivity (0–1) |
| `tempo` | BPM |
| `danceability` | Suitability for dancing (0–1) |
| `acousticness` | Confidence of acoustic sound |
| `instrumentalness` | Probability of no vocals |
| `liveness` | Presence of live audience |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/[your-username]/vibecheck-ai.git
cd vibecheck-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Or explore the notebook

```bash
jupyter notebook notebook.ipynb
```

---

## 🗂️ Project Structure

```
vibecheck-ai/
├── app.py                  # Streamlit interactive app
├── notebook.ipynb          # Full analysis notebook
├── requirements.txt
├── src/
│   ├── generate_data.py    # Synthetic Spotify-like dataset generator
│   └── ml_pipeline.py      # ML models (clustering, neural net, NLP)
├── plots/                  # Generated visualizations
└── data/                   # Dataset (auto-generated)
```

---

## 📸 Screenshots

### Vibe Map — Energy × Valence
Tracks plotted in 2D space, colored by discovered vibe cluster.

### Neural Network Training
Loss curves showing train/validation convergence over 60 epochs.

### Vibe Radar Chart
Comparison of audio feature profiles across all 5 vibe clusters.

---

## 🔌 Use Real Spotify Data

This project ships with a synthetic dataset. To use real Spotify data:

1. Create an app at [developer.spotify.com](https://developer.spotify.com)
2. Install `spotipy`: `pip install spotipy`
3. Replace `generate_dataset()` calls with your Spotify API calls

---

## 🛣️ Roadmap

- [ ] Integrate real Spotify API via `spotipy`
- [ ] UMAP for better cluster visualization
- [ ] DistilBERT for richer NLP on track names
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Add DJ set sequence recommender

---

## 📄 License

MIT — feel free to use, fork, and build on this.

---

*Built by [Your Name] · [LinkedIn](https://linkedin.com) · [Portfolio](https://yoursite.com)*
