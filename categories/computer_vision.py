import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "Sürücü Uyuklama Tespiti",
        "Maske Kullanımı Tespiti",
        "El İşaretleri ve Parmak Sayma"
    ])

    # ----------------- 1. SÜRÜCÜ UYUKLAMA TESPİTİ -----------------
    if project == "Sürücü Uyuklama Tespiti":
        st.header("👁️ Sürücü Uyuklama / Yorgunluk Tespiti")
        st.write("Fotoğraf yükleyerek göz durumunu (Açık/Kapalı) analiz edin.")

        show_dataset_info("drowsy")

        with st.expander("⚠️ Bu Modül Nasıl Çalışıyor? (Not: Şu An Simülasyon)"):
            st.code(TRAIN_CODE["drowsy"], language="python")

        show_model_metrics("drowsy")

        uploaded_file = st.file_uploader("Bir Sürücü Fotoğrafı Yükleyin", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Resim", width=400)

            # Görsel işleme simülasyonu (EAR - Eye Aspect Ratio mantığı)
            # Normal şartlarda dlib veya mediapipe mesh ile göz kırpması sayılır
            st.info("🔄 Göz noktaları taranıyor...")

            # Örnek bir mantık çalıştırma (Proje iskeleti için)
            result = np.random.choice(["Uyanık - Yolculuğa Hazır ✅", "Yorgun / Uykulu - Mola Verilmeli! ⚠️"])
            if "Uyanık" in result:
                st.success(f"Durum: {result}")
            else:
                st.error(f"Durum: {result}")

    # ----------------- 2. MASKE KULLANIMI TESPİTİ -----------------
    elif project == "Maske Kullanımı Tespiti":
        st.header("😷 Maske Kullanımı Tespiti")
        st.write("Yüz fotoğrafı yükleyin ve maske olup olmadığını kontrol edin.")

        show_dataset_info("mask")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["mask"], language="python")

        show_model_metrics("mask")

        uploaded_file = st.file_uploader("Yüz Fotoğrafı Seçin", type=["jpg", "jpeg", "png"], key="mask")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Analiz Edilen Resim", width=400)

            try:
                # Resmi OpenCV formatına çevirme ve ön işleme
                img = np.array(image)
                img_resized = cv2.resize(img, (64, 64))

                # Eğer resim RGBA ise RGB'ye çevir
                if img_resized.shape[-1] == 4:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
                elif len(img_resized.shape) == 2:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

                flattened_img = img_resized.flatten().reshape(1, -1)

                # Modeli yükle ve tahmin et
                model = load_model('mask_model.pkl')
                # Modelimiz 0: Maskeli, 1: Maskesiz olarak eğitildi
                pred = model.predict(flattened_img)[0]

                if pred == 0:
                    st.success("😷 Maske Tespit Edildi. Güvenli Bölge!")
                else:
                    st.error("⚠️ Maske Bulunamadı! Lütfen maskenizi takın.")
            except Exception as e:
                st.warning("Model dosyası eksik veya resim formatı uyumsuz. Simülasyon modu aktif.")
                st.info(f"Tahmin: {np.random.choice(['Maskeli ✅', 'Maskesiz ❌'])}")

    # ----------------- 3. EL İŞARETLERİ VE PARMAK SAYMA -----------------
    elif project == "El İşaretleri ve Parmak Sayma":
        st.header("✌️ El İşaretleri ve Parmak Sayma")
        st.write("Yüklediğiniz el resmindeki eklem noktalarını (MediaPipe) çıkartır ve parmak sayar.")

        show_dataset_info("hand")

        with st.expander("ℹ️ Bu Modül Nasıl Çalışıyor? (Pretrained Model)"):
            st.code(TRAIN_CODE["hand"], language="python")

        show_model_metrics("hand")

        uploaded_file = st.file_uploader("Net bir el fotoğrafı yükleyin", type=["jpg", "jpeg", "png"], key="hand")

        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file))

            # MediaPipe Hands Başlatma
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils

            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                # MediaPipe RGB resim ister
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    st.success("🖐️ El eklem noktaları başarıyla tespit edildi!")
                    annotated_image = image.copy()

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Eklem noktalarını resmin üzerine çizme
                        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Basit parmak sayma algoritması (Baş parmak hariç uç noktaların alt noktalardan yukarıda olması durumu)
                        # İşaret, Orta, Yüzük, Serçe parmak uçları: 8, 12, 16, 20
                        finger_ids = [8, 12, 16, 20]
                        opened_fingers = 0
                        for f_id in finger_ids:
                            if hand_landmarks.landmark[f_id].y < hand_landmarks.landmark[f_id - 2].y:
                                opened_fingers += 1

                    st.image(annotated_image, caption=f"Analiz Sonucu: Yaklaşık {opened_fingers} parmak açık!", width=500)
                else:
                    st.warning("Resimde net bir el bulunamadı. Lütfen elinizin kameraya düz baktığı bir fotoğraf seçin.")
                    st.image(image, width=400)
