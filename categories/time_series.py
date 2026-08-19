import streamlit as st
import numpy as np
import pandas as pd

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "Hisse Senedi Fiyat Tahmini",
        "Hava Durumu / Sıcaklık Tahmini",
        "Mağaza Satış Tahmini"
    ])

    # ----------------- 1. HİSSE SENEDİ FİYAT TAHMİNİ -----------------
    if project == "Hisse Senedi Fiyat Tahmini":
        st.header("📈 Hisse Senedi Fiyat Tahmini (Apple - AAPL)")
        st.write("Apple hissesinin geçmiş verilerine dayanarak önümüzdeki 7 günün fiyat trendini simüle edin.")

        show_dataset_info("stock")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["stock"], language="python")

        show_model_metrics("stock")

        days_to_predict = st.slider("Tahmini Gün Sayısı", 1, 14, 7)

        if st.button("Gelecek Trendi Tahmin Et"):
            try:
                model = load_model('stock_model.pkl')
                recent_data = load_model('stock_recent.pkl')  # Son 30 gunun kapanis fiyatlari

                # Gelecek günleri tahmin etme döngüsü
                predictions = []
                current_input = recent_data[-1]

                for _ in range(days_to_predict):
                    pred = model.predict(np.array([[current_input]]))[0]
                    predictions.append(pred)
                    current_input = pred

                # Grafik için veri birleştirme
                history_df = pd.DataFrame({'Fiyat ($)': recent_data, 'Tür': 'Geçmiş Değerler'})
                future_df = pd.DataFrame({'Fiyat ($)': predictions, 'Tür': 'Gelecek Tahminleri'})
                total_df = pd.concat([history_df, future_df]).reset_index(drop=True)

                # Grafik Çizimi
                st.write("📊 **Son 30 Gün ve Gelecek Tahmin Grafiği**")
                st.line_chart(total_df['Fiyat ($)'])
                st.success(f"🚀 {days_to_predict} gün sonraki tahmini kapanış fiyatı: **${predictions[-1]:.2f}**")
            except FileNotFoundError:
                st.warning("Model dosyaları eksik. Lütfen önce 'train_timeseries.py' dosyasını çalıştırın.")

    # ----------------- 2. HAVA DURUMU / SICAKLIK TAHMİNİ -----------------
    elif project == "Hava Durumu / Sıcaklık Tahmini":
        st.header("🌦️ Hava Durumu / Ortalama Sıcaklık Tahmini")
        st.write("Geçmiş iklim verilerinden yola çıkarak önümüzdeki günlerin sıcaklık değişimini öngörün.")

        show_dataset_info("weather")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["weather"], language="python")

        show_model_metrics("weather")

        days_to_predict = st.slider("Tahmin Edilecek Gün Sayısı", 1, 10, 5, key="weather_days")

        if st.button("Sıcaklık Trendini Hesapla"):
            try:
                model = load_model('weather_model.pkl')
                recent_data = load_model('weather_recent.pkl')

                predictions = []
                current_input = recent_data[-1]
                for _ in range(days_to_predict):
                    pred = model.predict(np.array([[current_input]]))[0]
                    predictions.append(pred)
                    current_input = pred

                history_df = pd.DataFrame({'Sıcaklık (°C)': recent_data})
                future_df = pd.DataFrame({'Sıcaklık (°C)': predictions})
                total_df = pd.concat([history_df, future_df]).reset_index(drop=True)

                st.line_chart(total_df['Sıcaklık (°C)'])
                st.info(f"🌡️ {days_to_predict} gün sonra tahmini ortalama sıcaklık: **{predictions[-1]:.1f}°C**")
            except FileNotFoundError:
                st.warning("Hava durumu model dosyaları eksik.")

    # ----------------- 3. MAĞAZA SATIŞ TAHMİNİ -----------------
    elif project == "Mağaza Satış Tahmini":
        st.header("🛒 Mağaza Satış Tahmini (Walmart Analitiği)")
        st.write("Haftalık ciro verilerini analiz ederek önümüzdeki haftaların ciro talebini tahmin edin.")

        show_dataset_info("walmart")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["walmart"], language="python")

        show_model_metrics("walmart")

        weeks_to_predict = st.slider("Tahmin Edilecek Hafta Sayısı", 1, 4, 2)

        if st.button("Haftalık Ciroyu Tahmin Et"):
            try:
                model = load_model('walmart_model.pkl')
                recent_data = load_model('walmart_recent.pkl')

                predictions = []
                current_input = recent_data[-1]
                for _ in range(weeks_to_predict):
                    pred = model.predict(np.array([[current_input]]))[0]
                    predictions.append(pred)
                    current_input = pred

                history_df = pd.DataFrame({'Haftalık Satış ($)': recent_data})
                future_df = pd.DataFrame({'Haftalık Satış ($)': predictions})
                total_df = pd.concat([history_df, future_df]).reset_index(drop=True)

                st.line_chart(total_df['Haftalık Satış ($)'])
                st.success(f"💳 {weeks_to_predict} hafta sonra beklenen tahmini mağaza cirosu: **${predictions[-1]:,.2f}**")
            except FileNotFoundError:
                st.warning("Walmart model dosyaları bulunamadı.")
