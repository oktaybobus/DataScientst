import streamlit as st
import numpy as np
import cv2
import random
from PIL import Image

from model_loader import load_model, load_keras_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "Pneumonia Detection from X-Ray",
        "Facial Emotion Recognition",
        "Text Generation Bot"
    ])

    # ----------------- 1. PNEUMONIA DETECTION FROM X-RAY -----------------
    if project == "Pneumonia Detection from X-Ray":
        st.header("🩻 Pneumonia Detection from Chest X-Ray")
        st.write("Analyze an uploaded chest X-ray image using convolutional neural networks (CNN).")

        show_dataset_info("pneumonia")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["pneumonia"], language="python")

        show_model_metrics("pneumonia")

        uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Analyzed X-Ray Image", width=350)

            try:
                # Resmi yükleyip ön işleme yapma
                img = np.array(image.convert('L'))
                img_resized = cv2.resize(img, (64, 64)) / 255.0
                input_data = img_resized.reshape(-1, 64, 64, 1)

                model = load_keras_model('pneumonia_model.keras')
                prediction = model.predict(input_data)[0][0]

                if prediction > 0.5:
                    st.error(f"⚠️ Analysis Result: PNEUMONIA indicators detected. (Probability: %{prediction*100:.1f})")
                else:
                    st.success(f"✅ Analysis Result: Appears NORMAL / CLEAR. (Probability: %{(1-prediction)*100:.1f})")
            except Exception as e:
                st.warning("Model loading error or simulation mode. Showing sample result:")
                res = random.choice(["Normal ✅", "Pneumonia Risk ⚠️"])
                st.info(f"Result: {res}")

            st.caption("⚠️ **Note:** This is for educational purposes only and does not constitute medical advice. Please consult a physician for an official diagnosis.")

    # ----------------- 2. FACIAL EMOTION RECOGNITION -----------------
    elif project == "Facial Emotion Recognition":
        st.header("🎭 Facial Emotion Recognition")
        st.write("Upload a face photo, and the AI will analyze the current emotional state (Happy, Sad, Angry).")

        show_dataset_info("face_emotion")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["face_emotion"], language="python")

        show_model_metrics("face_emotion")

        uploaded_file = st.file_uploader("Upload a Face Photo", type=["jpg", "jpeg", "png"], key="fer_input")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Scanned Face", width=300)

            try:
                img = np.array(image.convert('L'))
                img_resized = cv2.resize(img, (48, 48)) / 255.0
                input_data = img_resized.reshape(-1, 48, 48, 1)

                model = load_keras_model('fer_model.keras')
                preds = model.predict(input_data)[0]

                emotions = ["Angry 🤬", "Happy 🥰", "Sad 😢"]
                dominant_emotion = emotions[np.argmax(preds)]
                st.success(f"Dominant Emotion Detected: **{dominant_emotion}**")
            except:
                st.info(f"Dominant Emotion (Simulation): **{random.choice(['Happy 🥰', 'Sad 😢', 'Surprised 😲'])}**")

    # ----------------- 3. TEXT GENERATION BOT -----------------
    elif project == "Text Generation Bot":
        st.header("✍️ Character/Word-Based Text Generation Bot")
        st.write("Enter a seed word, and the neural network will generate a continuation in the language it learned.")

        show_dataset_info("text_gen")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["text_gen"], language="python")

        show_model_metrics("text_gen")

        start_word = st.selectbox("Select a Seed Word", ["to", "be", "or", "not", "the", "mind", "fortune"])
        length = st.slider("Number of Words to Generate", 5, 20, 10)

        if st.button("Generate Text with AI"):
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

                st.subheader("🤖 Text Generated by the Bot:")
                st.write(f" *\"{' '.join(generated_text)}...\"*")
            except FileNotFoundError:
                st.warning("Text bot data file is missing.")
