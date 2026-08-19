import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "NBA Oyuncu Performans Gruplaması",
        "Kredi Kartı Müşteri Segmentasyonu",
        "Spotify Şarkı Tarzı Kümeleme"
    ])

    # ----------------- 1. NBA OYUNCU PERFORMANS GRUPLAMASI -----------------
    if project == "NBA Oyuncu Performans Gruplaması":
        st.header("🏆 NBA Oyuncu Performans Gruplaması (Sports Analytics)")
        st.write("Oyuncunun istatistiklerini girerek hangi performans kümesine ait olduğunu öğrenin.")

        show_dataset_info("nba")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["nba"], language="python")

        show_model_metrics("nba")

        col1, col2 = st.columns([1, 2])
        with col1:
            pts = st.slider("Maç Başına Sayı (Points)", 0.0, 35.0, 15.0)
            reb = st.slider("Maç Başına Ribaund", 0.0, 15.0, 5.0)
            ast = st.slider("Maç Başına Asist", 0.0, 12.0, 4.0)

            if st.button("Oyuncu Stilini Grupla"):
                model = joblib.load('models/nba_model.pkl')
                scaler = joblib.load('models/nba_scaler.pkl')

                # Girdiyi ölçeklendir ve tahmin et
                user_data = np.array([[pts, reb, ast]])
                user_scaled = scaler.transform(user_data)
                cluster_pred = model.predict(user_scaled)[0]

                styles = {
                    0: "⭐ Çok Yönlü Yıldız (All-Round Star)",
                    1: "🛡️ Ribaund ve Savunma Uzmanı (Big Man / Defender)",
                    2: "🏀 Skorer Oyun Kurucu (Scoring Guard)",
                    3: "🏃 Rol Oyuncusu / Benç Destek (Role Player)"
                }
                st.success(f"Bu istatistiklere sahip oyuncunun stili: **{styles[cluster_pred]}**")

        with col2:
            st.write("📊 **Performans Kümeleri Dağılımı (3D Grafiği)**")
            # Örnek görselleştirme için rastgele veri simülasyonu
            chart_data = pd.DataFrame(np.random.rand(100, 3) *30, columns=['Sayı', 'Ribaund', 'Asist'])
            chart_data['Küme'] = np.random.choice(['Grup 0', 'Grup 1', 'Grup 2', 'Grup 3'], size=100)

            fig = px.scatter_3d(chart_data, x='Sayı', y='Ribaund', z='Asist', color='Küme', opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, width="stretch")

    # ----------------- 2. KREDI KARTI MÜŞTERİ SEGMENTASYONU -----------------
    elif project == "Kredi Kartı Müşteri Segmentasyonu":
        st.header("💳 Kredi Kartı Kullanıcı Analizi")

        show_dataset_info("creditcard")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["creditcard"], language="python")

        show_model_metrics("creditcard")

        col1, col2 = st.columns([1, 2])
        with col1:
            balance = st.number_input("Mevcut Hesap Bakiyesi ($)", value=1500.0)
            purchases = st.number_input("Toplam Alışveriş Tutarı ($)", value=500.0)
            limit = st.number_input("Kredi Kartı Limiti ($)", value=4000.0)

            if st.button("Müşteriyi Segmentlere Ayır"):
                model = joblib.load('models/cc_model.pkl')
                scaler = joblib.load('models/cc_scaler.pkl')

                user_data = np.array([[balance, purchases, limit]])
                user_scaled = scaler.transform(user_data)
                cluster_pred = model.predict(user_scaled)[0]

                cc_profiles = {
                    0: "🛍️ Aktif ve Yüksek Harcamalı Müşteri",
                    1: "💵 Dengeli / Standart Kullanıcı",
                    2: "⚠️ Borç Eğilimli / Düşük Limitli Kullanıcı",
                    3: "💎 Varlıklı / Yüksek Limitli Az Harcayan"
                }
                st.info(f"Müşteri Segment Profili: **{cc_profiles[cluster_pred]}**")

        with col2:
            st.write("📊 **Finansal Segmentasyon Grafiği**")
            chart_data = pd.DataFrame(np.random.rand(100, 3) *30, columns=['Bakiye', 'Alışveriş', 'Limit'])
            chart_data['Segment'] = np.random.choice(['Segment 0', 'Segment 1', 'Segment 2', 'Segment 3'], size=100)
            fig = px.scatter_3d(chart_data, x='Bakiye', y='Alışveriş', z='Limit', color='Segment', opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, width="stretch")

    # ----------------- 3. SPOTIFY ŞARKI TARZI KÜMELEME -----------------
    elif project == "Spotify Şarkı Tarzı Kümeleme":
        st.header("🎵 Spotify Müzik Karakteristiği Kümeleme")

        show_dataset_info("spotify")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["spotify"], language="python")

        show_model_metrics("spotify")

        col1, col2 = st.columns([1, 2])
        with col1:
            dance = st.slider("Dans Edilebilirlik (Danceability)", 0.0, 1.0, 0.6)
            energy = st.slider("Enerji Seviyesi (Energy)", 0.0, 1.0, 0.7)
            loudness = st.slider("Ses Seviyesi (Loudness dB)", -60.0, 0.0, -8.0)

            if st.button("Şarkı Türü Kümesini Bul"):
                model = joblib.load('models/spotify_model.pkl')
                scaler = joblib.load('models/spotify_scaler.pkl')

                user_data = np.array([[dance, energy, loudness]])
                user_scaled = scaler.transform(user_data)
                cluster_pred = model.predict(user_scaled)[0]

                music_genres = {
                    0: "🔥 Enerjik / Kulüp ve Parti Şarkısı",
                    1: "☕ Akustik / Sakin ve Dinlendirici",
                    2: "🎸 Yoğun Sesli / Rock ve Metal Melodiler",
                    3: "📻 Popüler / Standart Radyo Şarkısı"
                }
                st.success(f"Şarkının Tahmini Atmosferi: **{music_genres[cluster_pred]}**")

        with col2:
            st.write("📊 **Müzik Evreni Kümeleme Haritası**")
            chart_data = pd.DataFrame(np.random.rand(100, 3) * [1, 1, -20], columns=['Dans', 'Enerji', 'Ses'])
            chart_data['Atmosfer'] = np.random.choice(['Tür 0', 'Tür 1', 'Tür 2', 'Tür 3'], size=100)
            fig = px.scatter_3d(chart_data, x='Dans', y='Enerji', z='Ses', color='Atmosfer', opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, width="stretch")
