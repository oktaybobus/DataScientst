import streamlit as st
import joblib

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "SMS Spam / Kimlik Avı Tespiti",
        "IMDb Film Yorumları Duygu Analizi",
        "Sahte Haber Tespiti"
    ])

    # ----------------- 1. SMS SPAM / KİMLİK AVI TESPİTİ -----------------
    if project == "SMS Spam / Kimlik Avı Tespiti":
        st.header("📧 SMS Spam / Kimlik Avı Tespiti")
        st.write("Gelen şüpheli mesajı kutuya yapıştırarak güvenli (Ham) mi yoksa dolandırıcılık (Spam) mı olduğunu analiz edin.")

        show_dataset_info("sms")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["sms"], language="python")

        show_model_metrics("sms")

        user_text = st.text_area("Mesaj Metni (İngilizce)", "Congratulations! You've won a $1,000 Walmart Gift Card. Click here to claim now.")

        if st.button("Mesajı Analiz Et"):
            try:
                # Model ve Vektörleştiriciyi Yükle
                model = joblib.load('models/spam_model.pkl')
                vectorizer = joblib.load('models/spam_vectorizer.pkl')

                # Metni sayısallaştır ve tahmin et
                text_vector = vectorizer.transform([user_text])
                pred = model.predict(text_vector)[0]

                if pred == 1:
                    st.error("⚠️ TESPİT: Bu mesaj büyük olasılıkla bir SPAM veya KİMLİK AVI (Phishing) girişimidir!")
                else:
                    st.success("✅ TESPİT: Bu mesaj GÜVENLİ (Ham) görünüyor.")
            except FileNotFoundError:
                st.warning("Model dosyaları bulunamadı. Lütfen önce 'train_nlp.py' dosyasını çalıştırın.")

    # ----------------- 2. IMDb FİLM YORUMLARI DUYGU ANALİZİ -----------------
    elif project == "IMDb Film Yorumları Duygu Analizi":
        st.header("🍿 IMDb Film Yorumları Duygu Analizi")
        st.write("Bir film hakkındaki eleştiriyi yazın, yapay zeka olumlu mu yoksa olumsuz mu olduğunu söylesin.")

        show_dataset_info("imdb_sentiment")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["imdb_sentiment"], language="python")

        show_model_metrics("imdb_sentiment")

        user_text = st.text_area("Film Eleştirisi (İngilizce)", "The movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.")

        if st.button("Duygu Durumunu Tahmin Et"):
            try:
                model = joblib.load('models/imdb_model.pkl')
                vectorizer = joblib.load('models/imdb_vectorizer.pkl')

                text_vector = vectorizer.transform([user_text])
                pred = model.predict(text_vector)[0]

                if pred == 1:
                    st.success("🥰 Sonuç: POZİTİF (Olumlu) bir eleştiri. İzleyiciler filmi sevmiş!")
                else:
                    st.error("🤬 Sonuç: NEGATİF (Olumsuz) bir eleştiri. İzleyiciler filmden hoşlanmamış.")
            except FileNotFoundError:
                st.warning("Model dosyaları bulunamadı. Lütfen önce modelleri eğitin.")

    # ----------------- 3. SAHTE HABER TESPİTİ -----------------
    elif project == "Sahte Haber Tespiti":
        st.header("📰 Sahte Haber Tespiti")
        st.write("Şüphelendiğiniz bir haber metnini veya makaleyi girerek gerçekliğini doğrulayın.")

        show_dataset_info("fake_news")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["fake_news"], language="python")

        show_model_metrics("fake_news")

        user_text = st.text_area("Haber İçeriği / Metni (İngilizce)", "The government has officially announced a new law that completely bans all social media platforms starting next Monday.")

        if st.button("Haber Gerçekliğini Sorgula"):
            try:
                model = joblib.load('models/news_model.pkl')
                vectorizer = joblib.load('models/news_vectorizer.pkl')

                text_vector = vectorizer.transform([user_text])
                pred = model.predict(text_vector)[0]

                if pred == 1:
                    st.error("🚨 DİKKAT: Bu haber büyük ihtimalle SAHTE (Uydurma/Manipülatif) bir içeriğe sahip!")
                else:
                    st.success("📰 SONUÇ: Bu haber GERÇEK ve doğrulanmış kaynaklara dayanıyor gibi görünüyor.")
            except FileNotFoundError:
                st.warning("Gerekli model dosyaları eksik.")
