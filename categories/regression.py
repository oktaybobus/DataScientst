import streamlit as st
import numpy as np
import pandas as pd

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", ["Altın Fiyatı Tahmini", "Öğrenci Sınav Puanı Tahmini", "Uber/Taksi Ücret Tahmini"])

    if project == "Altın Fiyatı Tahmini":
        st.header("🪙 Küresel Endekslerle Altın Fiyatı Tahmini")
        st.info(
            "Bu model, S&P 500 endeksi, ham petrol fiyatı (USO), gümüş fiyatı (SLV) ve "
            "EUR/USD paritesi gibi küresel finansal göstergeleri kullanarak altın ons fiyatını tahmin eder."
        )

        show_dataset_info("gold")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["gold"], language="python")

        show_model_metrics("gold")


        col1, col2 = st.columns(2)
        with col1:
            spx = st.number_input("S&P 500 Endeksi (SPX)", value=2200.0)
            uso = st.number_input("Ham Petrol Fiyatı (USO)", value=40.0)
        with col2:
            slv = st.number_input("Gümüş Fiyatı (SLV)", value=15.0)
            eur_usd = st.number_input("EUR/USD Paritesi", value=1.12)
        if st.button("Altın Fiyatını Tahmin Et"):
            model = load_model('gold_model.pkl')
            pred = model.predict(np.array([[spx, uso, slv, eur_usd]]))
            st.success(f"💰 Tahmini Altın Ons Fiyatı: **${pred.item():,.2f}**")


# ----------------- 2. ÖĞRENCİ SINAV PUANI TAHMİNİ -----------------
    elif project == "Öğrenci Sınav Puanı Tahmini":
        st.header("📝 Öğrenci Matematik Sınav Puanı Tahmini")
        st.write("Öğrencinin sosyal durumuna ve diğer sınav notlarına göre matematik puanını tahmin edin.")

        show_dataset_info("student")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["student"], language="python")

        show_model_metrics("student")

        col1, col2 = st.columns(2)
        with col1:
            reading = st.slider("Okuma Notu (Reading Score)", 0, 100, 70)
            writing = st.slider("Yazma Notu (Writing Score)", 0, 100, 70)
            gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
            prep = st.selectbox("Hazırlık Kursu Tamamlandı mı?", ["Hayır", "Evet"])
        with col2:
            lunch = st.selectbox("Öğle Yemeği Tipi", ["Standart", "Ücretsiz/İndirimli"])
            race = st.selectbox("Etnik Köken / Grup", ["Group A", "Group B", "Group C", "Group D", "Group E"])
            edu = st.selectbox("Ebeveyn Eğitim Seviyesi", [
                "some high school", "high school", "some college",
                "associate's degree", "bachelor's degree", "master's degree"
            ])

        if st.button("Matematik Notunu Tahmin Et"):
            try:
                model = load_model('student_model.pkl')

                # One-hot encoding eğitim sütun sırasına göre girdileri True/False (1/0) olarak hazırlama
                g_male = 1 if gender == "Erkek" else 0
                r_B = 1 if race == "Group B" else 0
                r_C = 1 if race == "Group C" else 0
                r_D = 1 if race == "Group D" else 0
                r_E = 1 if race == "Group E" else 0

                e_asc = 1 if edu == "associate's degree" else 0
                e_bach = 1 if edu == "bachelor's degree" else 0
                e_hs = 1 if edu == "high school" else 0
                e_mast = 1 if edu == "master's degree" else 0
                e_sc = 1 if edu == "some college" else 0

                l_std = 1 if lunch == "Standart" else 0
                p_none = 1 if prep == "Hayır" else 0

                # Model eğitimindeki tam sütun dizilimi:
                features = np.array([[
                    reading, writing, g_male, r_B, r_C, r_D, r_E,
                    e_asc, e_bach, e_hs, e_mast, e_sc, l_std, p_none
                ]])

                pred = model.predict(features)
                st.success(f"🎯 Tahmini Matematik Sınav Notu: **{int(pred[0])} / 100**")
            except FileNotFoundError:
                st.error("Hata: 'models/student_model.pkl' dosyası bulunamadı. Lütfen önce modeli eğitin.")

    # ----------------- 3. UBER/TAKSİ ÜCRET TAHMİNİ -----------------
    elif project == "Uber/Taksi Ücret Tahmini":
        st.header("🚗 Uber/Taksi Yolculuk Ücreti Tahmini")
        st.write("Alış ve varış koordinatları ile yolcu sayısına göre tahmini yolculuk maliyetini hesaplayın.")

        show_dataset_info("uber")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["uber"], language="python")

        show_model_metrics("uber")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Rota Koordinatları")
            # Varsayılan olarak popüler bir New York rotası koordinatları girilmiştir
            p_lat = st.number_input("Alış Enlemi (Pickup Latitude)", value=40.7128, format="%.4f")
            p_lon = st.number_input("Alış Boylamı (Pickup Longitude)", value=-74.0060, format="%.4f")
            d_lat = st.number_input("Varış Enlemi (Dropoff Latitude)", value=40.7829, format="%.4f")
            d_lon = st.number_input("Varış Boylamı (Dropoff Longitude)", value=-73.9654, format="%.4f")
            passengers = st.slider("Yolcu Sayısı", 1, 6, 1)

        with col2:
            st.subheader("🗺️ Canlı Rota Görünümü")
            # Haritada noktaları göstermek için veri çerçevesi (DataFrame) oluşturuyoruz
            map_data = pd.DataFrame({
                'lat': [p_lat, d_lat],
                'lon': [p_lon, d_lon]
            })
            st.map(map_data)

        if st.button("Yolculuk Ücretini Hesapla"):
            try:
                model = load_model('uber_model.pkl')

                # Model eğitimindeki tam sütun dizilimi:
                # pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count
                features = np.array([[p_lon, p_lat, d_lon, d_lat, passengers]])

                pred = model.predict(features)
                st.success(f"💳 Tahmini Taksimetre / Uber Ücreti: **${pred[0]:,.2f}**")
            except FileNotFoundError:
                st.error("Hata: 'models/uber_model.pkl' dosyası bulunamadı. Lütfen önce modeli eğitin.")
