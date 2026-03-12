"""
VibeCheck AI — Streamlit App
Interactive DJ music analysis tool.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.generate_data import generate_dataset
from src.ml_pipeline import (
    preprocess, train_clustering, find_optimal_k,
    train_neural_network, analyze_sentiment_simple,
    get_vibe_label, AUDIO_FEATURES
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeCheck AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e0e0e; }
    .stApp { background: linear-gradient(135deg, #0e0e0e 0%, #1a0a2e 100%); }
    h1, h2, h3 { color: #e0b0ff; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(224,176,255,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎧 VibeCheck AI")
st.markdown("*AI-powered music mood & energy analyzer for DJs — built with Machine Learning*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    n_samples = st.slider("Dataset size", 1000, 10000, 5000, 500)
    n_clusters = st.slider("Number of vibe clusters", 3, 8, 5)
    st.divider()
    st.markdown("### 🎚️ Filter tracks")
    energy_range = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), 0.05)
    valence_range = st.slider("Valence (positivity)", 0.0, 1.0, (0.0, 1.0), 0.05)
    tempo_range = st.slider("Tempo (BPM)", 80, 180, (80, 180))

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(n):
    return generate_dataset(n)

@st.cache_resource
def run_clustering(n, k):
    df = load_data(n)
    X, scaler = preprocess(df)
    kmeans, labels, sil = train_clustering(X, k)
    df["cluster"] = labels
    df["vibe_label"] = df["cluster"].apply(get_vibe_label)
    df["sentiment"] = analyze_sentiment_simple(df["track_name"])
    return df, kmeans, scaler, sil

with st.spinner("🔄 Training ML models..."):
    df, kmeans, scaler, sil_score = run_clustering(n_samples, n_clusters)

# Apply filters
mask = (
    df["energy"].between(*energy_range) &
    df["valence"].between(*valence_range) &
    df["tempo"].between(*tempo_range)
)
df_filtered = df[mask]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Vibe Map", "📊 Analysis", "🧠 Neural Net", "🔍 Predict"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Vibe Map
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🗺️ Vibe Clusters — Energy × Valence")
    st.markdown(f"*{len(df_filtered):,} tracks · Silhouette score: **{sil_score:.3f}***")

    col1, col2, col3, col4 = st.columns(4)
    for col, feat, emoji in zip(
        [col1, col2, col3, col4],
        ["energy", "danceability", "valence", "tempo"],
        ["⚡", "💃", "😊", "🥁"]
    ):
        col.metric(f"{emoji} Avg {feat.capitalize()}", f"{df_filtered[feat].mean():.2f}")

    st.divider()

    fig = px.scatter(
        df_filtered.sample(min(2000, len(df_filtered))),
        x="valence", y="energy",
        color="vibe_label",
        size="danceability",
        hover_data=["track_name", "genre", "tempo"],
        title="Vibe Clusters — Energy × Valence",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark",
        opacity=0.7,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution per cluster
    cluster_counts = df_filtered["vibe_label"].value_counts().reset_index()
    cluster_counts.columns = ["Vibe", "Count"]
    fig2 = px.bar(
        cluster_counts, x="Vibe", y="Count",
        color="Vibe",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark",
        title="Track distribution per vibe"
    )
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Audio Features Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Radar chart per vibe
        features_radar = ["energy", "valence", "danceability", "acousticness", "liveness"]
        vibe_means = df_filtered.groupby("vibe_label")[features_radar].mean().reset_index()

        fig_radar = go.Figure()
        colors = px.colors.qualitative.Vivid
        for i, row in vibe_means.iterrows():
            values = [row[f] for f in features_radar]
            values.append(values[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=features_radar + [features_radar[0]],
                fill="toself",
                name=row["vibe_label"],
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark",
            title="Feature Radar by Vibe",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        # Correlation heatmap
        corr = df_filtered[AUDIO_FEATURES].corr()
        fig_heat = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            title="Audio Features Correlation",
            template="plotly_dark",
            zmin=-1, zmax=1
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=400
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Tempo distribution
    fig_tempo = px.histogram(
        df_filtered, x="tempo", color="vibe_label",
        nbins=50, barmode="overlay", opacity=0.7,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title="BPM Distribution by Vibe"
    )
    fig_tempo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_tempo, use_container_width=True)

    # NLP Sentiment
    st.markdown("### 🧬 NLP Sentiment Analysis on Track Names")
    st.markdown("Keyword-based sentiment scoring: 1.0 = positive, 0.0 = negative")
    fig_sent = px.box(
        df_filtered, x="vibe_label", y="sentiment", color="vibe_label",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title="Track Name Sentiment by Vibe Cluster"
    )
    fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sent, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Neural Network
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🧠 Deep Learning — Danceability Predictor")
    st.markdown("A neural network trained to predict danceability from other audio features.")

    if st.button("🚀 Train Neural Network"):
        with st.spinner("Training... (~30 sec)"):
            model, history, metrics = train_neural_network(df)

        col1, col2 = st.columns(2)
        col1.metric("📉 MAE", f"{metrics['mae']:.4f}")
        col2.metric("📈 R² Score", f"{metrics['r2']:.4f}")

        # Training curves
        hist_df = pd.DataFrame({
            "epoch": range(1, len(history.history["loss"]) + 1),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        })
        fig_hist = px.line(
            hist_df, x="epoch", y=["train_loss", "val_loss"],
            template="plotly_dark",
            title="Training vs Validation Loss",
            color_discrete_sequence=["#a855f7", "#38bdf8"]
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Predicted vs Actual
        fig_pred = px.scatter(
            x=metrics["y_test"][:500], y=metrics["y_pred"][:500],
            labels={"x": "Actual Danceability", "y": "Predicted Danceability"},
            template="plotly_dark",
            title="Predicted vs Actual Danceability",
            opacity=0.6,
            color_discrete_sequence=["#a855f7"]
        )
        fig_pred.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(color="white", dash="dash"))
        fig_pred.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("👆 Click the button to train the neural network (may take ~30 sec)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predict a track
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔍 Predict the Vibe of a Track")
    st.markdown("Adjust the sliders to match your track's audio features — the model will classify its vibe.")

    col1, col2, col3 = st.columns(3)
    with col1:
        inp_energy = st.slider("⚡ Energy", 0.0, 1.0, 0.75)
        inp_valence = st.slider("😊 Valence", 0.0, 1.0, 0.50)
        inp_dance = st.slider("💃 Danceability", 0.0, 1.0, 0.80)
    with col2:
        inp_tempo = st.slider("🥁 Tempo (BPM)", 80, 180, 130)
        inp_loud = st.slider("🔊 Loudness (dB)", -20.0, 0.0, -7.0)
        inp_acoustic = st.slider("🎸 Acousticness", 0.0, 1.0, 0.05)
    with col3:
        inp_instrumental = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.85)
        inp_live = st.slider("🎤 Liveness", 0.0, 1.0, 0.15)
        inp_speech = st.slider("💬 Speechiness", 0.0, 0.5, 0.04)

    if st.button("🎯 Predict Vibe"):
        input_vec = np.array([[
            inp_energy, inp_valence, inp_tempo, inp_dance,
            inp_acoustic, inp_instrumental, inp_live, inp_loud, inp_speech
        ]])
        input_scaled = scaler.transform(input_vec)
        cluster_id = kmeans.predict(input_scaled)[0]
        vibe = get_vibe_label(cluster_id)

        st.markdown(f"### Predicted Vibe: **{vibe}**")

        # Show similar tracks
        similar = df[df["cluster"] == cluster_id].sample(min(10, len(df[df["cluster"] == cluster_id])))
        st.markdown("#### 🎵 Similar tracks in your dataset:")
        st.dataframe(
            similar[["track_name", "genre", "energy", "valence", "tempo", "danceability"]].reset_index(drop=True),
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>VibeCheck AI · Built with Python, Scikit-learn, TensorFlow & Streamlit</small></center>",
    unsafe_allow_html=True
)
