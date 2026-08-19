import streamlit as st

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Ajan Senaryosu Seçin", [
        "Akıllı Tarım ve Sulama Ajanı",
        "SSS Destek ve Niyet Analizi Ajanı",
        "Otonom Araç Simülasyon Ajanı"
    ])

    # ----------------- 1. AKILLI TARIM VE SULAMA AJANI -----------------
    if project == "Akıllı Tarım ve Sulama Ajanı":
        st.header("🌾 Akıllı Tarım ve Sulama Karar Ajanı")
        st.write("Ajan; toprak nemi, sıcaklık ve güneş ışığı verilerini anlık izleyerek su israfını önleyecek otonom kararlar alır.")

        show_dataset_info("farm_agent")

        with st.expander("⚙️ Ajanın Karar Mantığını Gör"):
            st.code(TRAIN_CODE["farm_agent"], language="python")

        show_model_metrics("farm_agent")

        col1, col2 = st.columns(2)
        with col1:
            soil_moisture = st.slider("Toprak Nem Oranı (%)", 0, 100, 25)
            temperature = st.slider("Hava Sıcaklığı (°C)", -10, 50, 32)
            sunlight = st.select_slider("Güneş Işığı Şiddeti", options=["Düşük", "Orta", "Yüksek"])

        with col2:
            st.subheader("🤖 Ajanın Otonom Karar Mekanizması")
            if st.button("Sensör Verilerini Analiz Et"):
                st.info("🔄 Tarım ajanı sensör loglarını okuyor ve buharlaşma riskini hesaplıyor...")

                # Otonom Karar Ağacı Mantığı
                if soil_moisture < 30 and temperature > 30 and sunlight == "Yüksek":
                    st.error("🚨 KARAR: Acil Yoğun Sulama Gerekli! (Buharlaşma riski yüksek, toprak kritik seviyede kuru).")
                    st.metric("Tavsiye Edilen Su Miktarı", "45 Litre / m²")
                elif soil_moisture < 40 and temperature > 15:
                    st.warning("⚠️ KARAR: Standart Sulama Başlatıldı. (Toprak nemi ideal sınırın altında).")
                    st.metric("Tavsiye Edilen Su Miktarı", "20 Litre / m²")
                else:
                    st.success("✅ KARAR: Sulama Gerekli Değil. (Toprak nemi yeterli, su tasarrufu modu aktif).")
                    st.metric("Tavsiye Edilen Su Miktarı", "0 Litre / m²")

    # ----------------- 2. SSS DESTEK VE NİYET ANALİZİ AJANI -----------------
    elif project == "SSS Destek ve Niyet Analizi Ajanı":
        st.header("💬 SSS Destek ve Niyet Analizi Ajanı")
        st.write("Kurumsal müşteri ajanı, gelen mesajın arkasındaki niyetini (Intent) otonom analiz ederek doğru aksiyonu tetikler.")

        show_dataset_info("faq_agent")

        with st.expander("⚙️ Ajanın Karar Mantığını Gör"):
            st.code(TRAIN_CODE["faq_agent"], language="python")

        show_model_metrics("faq_agent")

        user_message = st.text_area("Müşteri Mesajı (Örnek metni değiştirebilirsiniz):",
                                    "Merhaba, 3 gün önce verdiğim sipariş hâlâ kargoya verilmedi. İptal edip paramı geri almak istiyorum.")

        if st.button("Mesajı Ajan ile İşle"):
            st.info("🔄 Ajan niyet analizi (Intent Classification) ve anahtar kelime taraması yapıyor...")

            # Basit kural tabanlı niyet yakalama simülasyonu
            msg_lower = user_message.lower()

            if "i̇ade" in msg_lower or "iptal" in msg_lower or "para" in msg_lower:
                intent = "🚨 Finans / İade ve İptal Talebi"
                action = "Müşterinin fatura geçmişi doğrulandı. İptal talebi otonom olarak Muhasebe departmanına aktarıldı ve iade süreci başlatıldı."
            elif "kargo" in msg_lower or "sipariş" in msg_lower or "nerede" in msg_lower:
                intent = "📦 Lojistik / Kargo ve Teslimat Takibi"
                action = "Sipariş numarası tespit edilmeye çalışılıyor. Sistem otonom olarak Yurtiçi Kargo API'sine sorgu gönderdi."
            else:
                intent = "💬 Genel / Teşekkür - Bilgi Talebi"
                action = "Mesaj standart SSS havuzuna yönlendirildi. Ajan otomatik yapay zeka cevabını hazırlıyor."

            st.subheader("🤖 Ajan Analiz Raporu:")
            st.write(f"**Tespit Edilen Niyet:** {intent}")
            st.success(f"**Otonom Alınan Aksiyon:** {action}")

    # ----------------- 3. OTONOM ARAÇ SİMÜLASYON AJANI -----------------
    elif project == "Otonom Araç Simülasyon Ajanı":
        st.header("🚗 Otonom Araç Şerit Takip ve Park Ajanı")
        st.write("Sanal araç ajanı, sensör girdilerini değerlendirerek şeritte kalma ve otonom park kararlarını simüle eder.")

        show_dataset_info("autonomous_car")

        with st.expander("⚙️ Ajanın Karar Mantığını Gör"):
            st.code(TRAIN_CODE["autonomous_car"], language="python")

        show_model_metrics("autonomous_car")

        sensor_distance = st.slider("Ön Araç ile Mesafe (Metre)", 1, 100, 15)
        lane_status = st.selectbox("Şerit Çizgisi Durumu", ["Net Görünür", "Kesikli / Silik", "Şerit Yok"])
        parking_slot = st.checkbox("Boş Park Yeri Tespit Edildi mi?")

        if st.button("Araç Ajanını Çalıştır"):
            st.subheader("🎬 Sürüş Esnasında Ajan Kararları:")

            # Şerit takip kararı
            if lane_status == "Net Görünür":
                st.success("🟢 ŞERİT TAKİP: Kameralar aktif. Şerit ortalanarak otonom sürüş güvenle sürdürülüyor.")
            else:
                st.warning("🟡 ŞERİT UYARISI: Şerit çizgileri yetersiz! Ajan, direksiyon kontrolünü sürücüye devretme hazırlığı yapıyor.")

            # Mesafe ve Fren kararı
            if sensor_distance < 20:
                st.error(f"🔴 ACİL FREN: Ön araçla mesafe {sensor_distance} metreye düştü! Güvenli takip mesafesi ihlali nedeniyle otonom fren yapıldı.")
            else:
                st.success("🟢 HIZ KONTROLÜ: Mesafe güvenli. Belirlenen hız sınırında otonom sürüşe devam ediliyor.")

            # Park Kararı
            if parking_slot and sensor_distance > 30:
                st.info("🔵 OTONOM PARK: Boş park yeri algılandı. Araç hızı düşürülüyor ve otonom dikey park algoritması başlatılıyor.")
