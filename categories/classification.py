import streamlit as st
import numpy as np

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", ["Mobil Cihaz Fiyat Segmenti", "Şarap Kalitesi Sınıflandırması", "Müşteri Terki Tahmini"])

    # ----------------- 1. MOBİL CİHAZ FİYAT SEGMENTİ -----------------
    if project == "Mobil Cihaz Fiyat Segmenti":
        st.header("📱 Mobil Cihaz Fiyat Segmenti Tahmini")
        st.write("Telefon özelliklerine göre hangi fiyat segmentine ait olduğunu tahmin edin.")

        show_dataset_info("mobile")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["mobile"], language="python")

        show_model_metrics("mobile")

        col1, col2 = st.columns(2)
        with col1:
            ram = st.slider("RAM Kapasitesi (MB)", 256, 4096, 2048)
            battery = st.slider("Batarya Gücü (mAh)", 500, 5000, 2000)
            int_mem = st.slider("Dahili Hafıza (GB)", 2, 64, 32)
            weight = st.number_input("Telefon Ağırlığı (Gram)", min_value=80, max_value=250, value=150)
            cores = st.selectbox("Çekirdek Sayısı (n_cores)", [1, 2, 4, 6, 8])
        with col2:
            clock = st.number_input("İşlemci Hızı (Clock Speed)", min_value=0.5, max_value=3.0, value=1.5)
            m_dep = st.number_input("Telefon Kalınlığı (m_dep)", min_value=0.1, max_value=1.0, value=0.5)
            pc = st.slider("Arka Kamera Çözünürlüğü (Megapiksel)", 0, 20, 10)
            sim = st.selectbox("Çift Hat Var mı?", ["Hayır", "Evet"])
            touch = st.selectbox("Dokunmatik Ekran mı?", ["Hayır", "Evet"])
            wifi = st.selectbox("Wi-Fi Var mı?", ["Hayır", "Evet"])

        if st.button("Fiyat Segmentini Tahmin Et"):
            model = load_model('mobile_model.pkl')
            sim_val = 1 if sim == "Evet" else 0
            touch_val = 1 if touch == "Evet" else 0
            wifi_val = 1 if wifi == "Evet" else 0

            features = np.array([[battery, clock, sim_val, int_mem, m_dep, weight, cores, pc, ram, touch_val, wifi_val]])
            pred = model.predict(features)[0]

            segments = {0: "📱 Ucuz Segment", 1: "💵 Normal Segment", 2: "💸 Pahalı Segment", 3: "🚀 Çok Pahalı / Amiral Gemisi"}
            st.success(f"Tahmini Sınıf: **{segments[pred]}**")

    # ----------------- 2. ŞARAP KALİTESİ SINIFLANDIRMASI -----------------
    elif project == "Şarap Kalitesi Sınıflandırması":
        st.header("🍷 Kırmızı Şarap Kalitesi Sınıflandırması")

        show_dataset_info("wine")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["wine"], language="python")

        show_model_metrics("wine")

        col1, col2 = st.columns(2)
        with col1:
            fixed_acid = st.number_input("Sabit Asitlik", value=8.3)
            vol_acid = st.number_input("Uçucu Asitlik", value=0.52)
            citric = st.number_input("Sitrik Asit", value=0.26)
            sugar = st.number_input("Kalan Şeker", value=2.2)
            chlorides = st.number_input("Klorürler", value=0.08)
        with col2:
            free_sulfur = st.number_input("Serbest Kükürt Dioksit", value=11.0)
            total_sulfur = st.number_input("Toplam Kükürt Dioksit", value=34.0)
            density = st.number_input("Yoğunluk (Density)", value=0.99)
            ph = st.number_input("pH Değeri", value=3.31)
            sulphates = st.number_input("Sülfatlar", value=0.56)
            alcohol = st.number_input("Alkol Oranı (%)", value=10.5)

        if st.button("Şarap Kalitesini Puanla"):
            model = load_model('wine_model.pkl')
            features = np.array([[fixed_acid, vol_acid, citric, sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]])
            pred = model.predict(features)[0]
            st.success(f"🎯 Tahmini Şarap Kalite Puanı: **{pred} / 10**")

    # ----------------- 3. MÜŞTERİ TERKİ TAHMİNİ -----------------
    elif project == "Müşteri Terki Tahmini":
        st.header("🚀 Telekom Müşteri Terki (Churn) Analizi")

        show_dataset_info("churn")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["churn"], language="python")

        show_model_metrics("churn")

        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Müşteri Kalma Süresi (Ay)", 0, 72, 12)
            monthly = st.number_input("Aylık Ödeme Tutarı ($)", value=65.0)
            total = st.number_input("Toplam Ödeme Tutarı ($)", value=780.0)
            gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        with col2:
            senior = st.selectbox("Yaşlı Vatandaş mı? (65+)", ["Hayır", "Evet"])
            partner = st.selectbox("Evli mi? (Partner)", ["Hayır", "Evet"])
            dependents = st.selectbox("Bakmakla Yükümlü Olduğu Biri Var mı?", ["Hayır", "Evet"])
            phone = st.selectbox("Telefon Servisi Var mı?", ["Hayır", "Evet"])

        if st.button("Terk Riskini Hesapla"):
            model = load_model('churn_model.pkl')
            g_val = 1 if gender == "Erkek" else 0
            s_val = 1 if senior == "Evet" else 0
            p_val = 1 if partner == "Evet" else 0
            d_val = 1 if dependents == "Evet" else 0
            ph_val = 1 if phone == "Evet" else 0

            features = np.array([[g_val, s_val, p_val, d_val, tenure, ph_val, monthly, total]])
            pred = model.predict(features)[0]

            if pred == 1:
                st.error("⚠️ Dikkat: Bu müşterinin şirketi **TERK ETME** riski yüksek!")
            else:
                st.success("✅ Güzel: Bu müşteri şirkette **KALMAYA** devam edecek gibi görünüyor.")
