import streamlit as st
import numpy as np
import cv2
from PIL import Image

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "Driver Drowsiness Detection",
        "Face Mask Detection",
        "Hand Gesture & Finger Counting"
    ])

    # ----------------- 1. DRIVER DROWSINESS DETECTION -----------------
    if project == "Driver Drowsiness Detection":
        st.header("👁️ Driver Drowsiness / Fatigue Detection")
        st.write("Upload a photo to analyze eye status (Open/Closed).")

        show_dataset_info("drowsy")

        with st.expander("⚠️ How Does This Module Work? (Note: Currently Simulation)"):
            st.code(TRAIN_CODE["drowsy"], language="python")

        show_model_metrics("drowsy")

        uploaded_file = st.file_uploader("Upload a Driver Photo", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)

            # Görsel işleme simülasyonu (EAR - Eye Aspect Ratio mantığı)
            st.info("🔄 Scanning eye landmarks...")

            result = np.random.choice(["Awake - Ready to Drive ✅", "Drowsy / Sleepy - Take a Break! ⚠️"])
            if "Awake" in result:
                st.success(f"Status: {result}")
            else:
                st.error(f"Status: {result}")

    # ----------------- 2. FACE MASK DETECTION -----------------
    elif project == "Face Mask Detection":
        st.header("😷 Face Mask Detection")
        st.write("Upload a face photo and check whether a mask is being worn.")

        show_dataset_info("mask")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["mask"], language="python")

        show_model_metrics("mask")

        uploaded_file = st.file_uploader("Select a Face Photo", type=["jpg", "jpeg", "png"], key="mask")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Analyzed Image", width=400)

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
                    st.success("😷 Mask Detected. Safe Zone!")
                else:
                    st.error("⚠️ No Mask Found! Please wear your mask.")
            except Exception as e:
                st.warning("Model file missing or image format incompatible. Simulation mode active.")
                st.info(f"Prediction: {np.random.choice(['With Mask ✅', 'Without Mask ❌'])}")

    # ----------------- 3. HAND GESTURE & FINGER COUNTING -----------------
    elif project == "Hand Gesture & Finger Counting":
        st.header("✌️ Hand Gesture & Finger Counting")
        st.write("Extracts hand joint landmarks (MediaPipe) from an uploaded image and counts fingers.")

        show_dataset_info("hand")

        with st.expander("ℹ️ How Does This Module Work? (Pretrained Model)"):
            st.code(TRAIN_CODE["hand"], language="python")

        show_model_metrics("hand")

        uploaded_file = st.file_uploader("Upload a clear hand photo", type=["jpg", "jpeg", "png"], key="hand")

        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file))

            if not HAS_MEDIAPIPE:
                st.warning("MediaPipe library is not available in this environment. Hand detection is disabled.")
                st.image(image, width=400)
                return

            # MediaPipe Hands Başlatma
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils

            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                # MediaPipe RGB resim ister
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    st.success("🖐️ Hand landmarks successfully detected!")
                    annotated_image = image.copy()

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Eklem noktalarını resmin üzerine çizme
                        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Basit parmak sayma algoritması
                        finger_ids = [8, 12, 16, 20]
                        opened_fingers = 0
                        for f_id in finger_ids:
                            if hand_landmarks.landmark[f_id].y < hand_landmarks.landmark[f_id - 2].y:
                                opened_fingers += 1

                    st.image(annotated_image, caption=f"Analysis Result: Approximately {opened_fingers} fingers open!", width=500)
                else:
                    st.warning("No clear hand found in the image. Please select a photo where your hand faces the camera directly.")
                    st.image(image, width=400)
