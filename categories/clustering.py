import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "NBA Player Performance Clustering",
        "Credit Card Customer Segmentation",
        "Spotify Song Style Clustering"
    ])

    # ----------------- 1. NBA PLAYER PERFORMANCE CLUSTERING -----------------
    if project == "NBA Player Performance Clustering":
        st.header("🏆 NBA Player Performance Clustering (Sports Analytics)")
        st.write("Enter a player's stats to find out which performance cluster they belong to.")

        show_dataset_info("nba")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["nba"], language="python")

        show_model_metrics("nba")

        col1, col2 = st.columns([1, 2])
        with col1:
            pts = st.slider("Points Per Game", 0.0, 35.0, 15.0)
            reb = st.slider("Rebounds Per Game", 0.0, 15.0, 5.0)
            ast = st.slider("Assists Per Game", 0.0, 12.0, 4.0)

            if st.button("Cluster Player Style"):
                model = load_model('nba_model.pkl')
                scaler = load_model('nba_scaler.pkl')

                # Girdiyi ölçeklendir ve tahmin et
                user_data = np.array([[pts, reb, ast]])
                user_scaled = scaler.transform(user_data)
                cluster_pred = model.predict(user_scaled)[0]

                styles = {
                    0: "⭐ All-Round Star",
                    1: "🛡️ Rebound & Defense Specialist (Big Man / Defender)",
                    2: "🏀 Scoring Guard",
                    3: "🏃 Role Player / Bench Support"
                }
                st.success(f"Player style based on these stats: **{styles[cluster_pred]}**")

        with col2:
            st.write("📊 **Performance Clusters Distribution (3D Chart)**")
            # Örnek görselleştirme için rastgele veri simülasyonu
            chart_data = pd.DataFrame(np.random.rand(100, 3) *30, columns=['Points', 'Rebounds', 'Assists'])
            chart_data['Cluster'] = np.random.choice(['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], size=100)

            fig = px.scatter_3d(chart_data, x='Points', y='Rebounds', z='Assists', color='Cluster', opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, width="stretch")

    # ----------------- 2. CREDIT CARD CUSTOMER SEGMENTATION -----------------
    elif project == "Credit Card Customer Segmentation":
        st.header("💳 Credit Card Customer Analysis")

        show_dataset_info("creditcard")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["creditcard"], language="python")

        show_model_metrics("creditcard")

        col1, col2 = st.columns([1, 2])
        with col1:
            balance = st.number_input("Current Account Balance ($)", value=1500.0)
            purchases = st.number_input("Total Purchase Amount ($)", value=500.0)
            limit = st.number_input("Credit Card Limit ($)", value=4000.0)

            if st.button("Segment Customer"):
                model = load_model('cc_model.pkl')
                scaler = load_model('cc_scaler.pkl')

                user_data = np.array([[balance, purchases, limit]])
                user_scaled = scaler.transform(user_data)
                cluster_pred = model.predict(user_scaled)[0]

                cc_profiles = {
                    0: "🛍️ Active / High Spender",
                    1: "💵 Balanced / Standard User",
                    2: "⚠️ Debt-Prone / Low Limit User",
                    3: "💎 Affluent / High Limit Low Spender"
                }
                st.info(f"Customer Segment Profile: **{cc_profiles[cluster_pred]}**")

        with col2:
            st.write("📊 **Financial Segmentation Chart**")
            chart_data = pd.DataFrame(np.random.rand(100, 3) *30, columns=['Balance', 'Purchases', 'Limit'])
            chart_data['Segment'] = np.random.choice(['Segment 0', 'Segment 1', 'Segment 2', 'Segment 3'], size=100)
            fig = px.scatter_3d(chart_data, x='Balance', y='Purchases', z='Limit', color='Segment', opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, width="stretch")

    # ----------------- 3. SPOTIFY SONG STYLE CLUSTERING -----------------
    elif project == "Spotify Song Style Clustering":
        st.header("🎵 Spotify Music Characteristics Clustering")

        show_dataset_info("spotify")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["spotify"], language="python")

        show_model_metrics("spotify")

        col1, col2 = st.columns([1, 2])
        with col1:
            dance = st.slider("Danceability", 0.0, 1.0, 0.6)
            energy = st.slider("Energy Level", 0.0, 1.0, 0.7)
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0)

            if st.button("Find Song Cluster"):
                model = load_model('spotify_model.pkl')
                scaler = load_model('spotify_scaler.pkl')

                user_data = np.array([[dance, energy, loudness]])
                user_scaled = scaler.transform(user_data)
                cluster_pred = model.predict(user_scaled)[0]

                music_genres = {
                    0: "🔥 Energetic / Club & Party Track",
                    1: "☕ Acoustic / Calm & Relaxing",
                    2: "🎸 Loud / Rock & Metal Melodies",
                    3: "📻 Popular / Standard Radio Hit"
                }
                st.success(f"Predicted Song Atmosphere: **{music_genres[cluster_pred]}**")

        with col2:
            st.write("📊 **Music Universe Clustering Map**")
            chart_data = pd.DataFrame(np.random.rand(100, 3) * [1, 1, -20], columns=['Dance', 'Energy', 'Loudness'])
            chart_data['Atmosphere'] = np.random.choice(['Type 0', 'Type 1', 'Type 2', 'Type 3'], size=100)
            fig = px.scatter_3d(chart_data, x='Dance', y='Energy', z='Loudness', color='Atmosphere', opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, width="stretch")
