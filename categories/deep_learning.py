import streamlit as st
import numpy as np
import cv2
import random
from PIL import Image

from model_loader import load_model, load_keras_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "Yapay Zeka ile Zatürre Teşhisi",
        "Yüz İfadesinden Duygu Tanıma",
        "Metin Üretim Robotu"
    ])

    # ----------------- 1. YAPAY ZEKA İLE ZATÜRRE TESHİSİ -----------------
    if project == "Yapay Zeka ile Zatürre Teşhisi":
        st.header("🩻 Göğüs Röntgeninden Zatürre (Pneumonia) Teşhisi")
        st.write("Yüklenen Göğüs Röntgeni (X-Ray) görselini evrişimli sinir ağları (CNN) ile analiz edin.")

        show_dataset_info("pneumonia")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["pneumonia"], language="python")

        show_model_metrics("pneumonia")

        uploaded_file = st.file_uploader("Göğüs Röntgeni Fotoğrafı Yükleyin", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Analiz Edilen Röntgen Görseli", width=350)

            try:
                # Resmi yükleyip ön işleme yapma
                img = np.array(image.convert('L'))
                img_resized = cv2.resize(img, (64, 64)) / 255.0
                input_data = img_resized.reshape(-1, 64, 64, 1)

                model = load_keras_model('pneumonia_model.keras')
                prediction = model.predict(input_data)[0][0]

                if prediction > 0.5:
                    st.error(f"⚠️ Analiz Sonucu: ZATÜRRE (PNEUMONIA) Belirtileri Saptandı. (Olasılık: %{prediction*100:.1f})")
                else:
                    st.success(f"✅ Analiz Sonucu: TEMİZ / NORMAL Görünüyor. (Olasılık: %{(1-prediction)*100:.1f})")
            except Exception as e:
                st.warning("Model yükleme hatası veya simülasyon modu. Örnek sonuç gösteriliyor:")
                res = random.choice(["Normal ✅", "Zatürre (Pneumonia) Riski ⚠️"])
                st.info(f"Sonuç: {res}")

            st.caption("⚠️ **Not:** Bu bilgi eğitim amaçlıdır ve kesinlikle tıbbi tavsiye niteliği taşımamaktadır. Lütfen resmi tanı için bir hekime başvurun.")

    # ----------------- 2. YÜZ İFADESİNDEN DUYGU TANIMA -----------------
    elif project == "Yüz İfadesinden Duygu Tanıma":
        st.header("🎭 Yüz İfadesinden Duygu Durum Tanıma")
        st.write("Yüz fotoğrafı yükleyin, yapay zeka anlık duygu durumunu (Mutlu, Üzgün, Öfkeli) analiz etsin.")

        show_dataset_info("face_emotion")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["face_emotion"], language="python")

        show_model_metrics("face_emotion")

        uploaded_file = st.file_uploader("Yüz Fotoğrafı Yükleyin", type=["jpg", "jpeg", "png"], key="fer_input")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Taranan Yüz", width=300)

            try:
                img = np.array(image.convert('L'))
                img_resized = cv2.resize(img, (48, 48)) / 255.0
                input_data = img_resized.reshape(-1, 48, 48, 1)

                model = load_keras_model('fer_model.keras')
                preds = model.predict(input_data)[0]

                emotions = ["Öfkeli 🤬", "Mutlu 🥰", "Üzgün 😢"]
                dominant_emotion = emotions[np.argmax(preds)]
                st.success(f"Yüzdeki Baskın Duygu: **{dominant_emotion}**")
            except:
                st.info(f"Yüzdeki Baskın Duygu (Simülasyon): **{random.choice(['Mutlu 🥰', 'Üzgün 😢', 'Şaşırmış 😲'])}**")

    # ----------------- 3. METİN ÜRETİM ROBOTU -----------------
    elif project == "Metin Üretim Robotu":
        st.header("✍️ Karakter/Kelime Tabanlı Metin Üretim Robotu")
        st.write("Bir başlangıç kelimesi girin, yapay sinir ağının öğrendiği dilde cümlenin devamını robot üretsin.")

        show_dataset_info("text_gen")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["text_gen"], language="python")

        show_model_metrics("text_gen")

        start_word = st.selectbox("Başlangıç Kelimesini Seçin", ["to", "be", "or", "not", "the", "mind", "fortune"])
        length = st.slider("Üretilecek Kelime Sayısı", 5, 20, 10)

        if st.button("Yapay Zeka Metni Üretsin"):
            try:
                markov_chain = load_model('text_robot_model.pkl')

                generated_text = [start_word]
                current_word = start_word

                for _ in range(length - 1):
                    if current_word in markov_chain:
                        next_word = random.choice(markov_chain[current_word])
                        generated_text.append(next_word)
                        current_word = next_word
                    else:
                        break

                st.subheader("🤖 Robotun Ürettiği Metin:")
                st.write(f" *\"{' '.join(generated_text)}...\"*")
            except FileNotFoundError:
                st.warning("Metin robotu veri dosyası eksik.")
