import streamlit as st

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "SMS Spam Detection",
        "IMDb Sentiment Analysis",
        "Fake News Detection"
    ])

    # ----------------- 1. SMS SPAM DETECTION -----------------
    if project == "SMS Spam Detection":
        st.header("📧 SMS Spam Detection")
        st.write("Paste a suspicious message to analyze whether it is safe (Ham) or fraudulent (Spam).")

        show_dataset_info("sms")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["sms"], language="python")

        show_model_metrics("sms")

        user_text = st.text_area("Message Text (English)", "Congratulations! You've won a $1,000 Walmart Gift Card. Click here to claim now.")

        if st.button("Analyze Message"):
            try:
                # Model ve Vektörleştiriciyi Yükle
                model = load_model('spam_model.pkl')
                vectorizer = load_model('spam_vectorizer.pkl')

                # Metni sayısallaştır ve tahmin et
                text_vector = vectorizer.transform([user_text])
                pred = model.predict(text_vector)[0]

                if pred == 1:
                    st.error("⚠️ DETECTED: This message is most likely SPAM or a PHISHING attempt!")
                else:
                    st.success("✅ DETECTED: This message appears to be SAFE (Ham).")
            except FileNotFoundError:
                st.warning("Model files not found. Please run 'train_nlp.py' first.")

    # ----------------- 2. IMDb SENTIMENT ANALYSIS -----------------
    elif project == "IMDb Sentiment Analysis":
        st.header("🍿 IMDb Movie Review Sentiment Analysis")
        st.write("Write a movie review, and the AI will determine whether it is positive or negative.")

        show_dataset_info("imdb_sentiment")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["imdb_sentiment"], language="python")

        show_model_metrics("imdb_sentiment")

        user_text = st.text_area("Movie Review (English)", "The movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.")

        if st.button("Predict Sentiment"):
            try:
                model = load_model('imdb_model.pkl')
                vectorizer = load_model('imdb_vectorizer.pkl')

                text_vector = vectorizer.transform([user_text])
                pred = model.predict(text_vector)[0]

                if pred == 1:
                    st.success("🥰 Result: POSITIVE review. The audience loved the movie!")
                else:
                    st.error("🤬 Result: NEGATIVE review. The audience did not enjoy the movie.")
            except FileNotFoundError:
                st.warning("Model files not found. Please train the models first.")

    # ----------------- 3. FAKE NEWS DETECTION -----------------
    elif project == "Fake News Detection":
        st.header("📰 Fake News Detection")
        st.write("Enter a suspicious news article or text to verify its authenticity.")

        show_dataset_info("fake_news")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["fake_news"], language="python")

        show_model_metrics("fake_news")

        user_text = st.text_area("News Content / Text (English)", "The government has officially announced a new law that completely bans all social media platforms starting next Monday.")

        if st.button("Check News Authenticity"):
            try:
                model = load_model('news_model.pkl')
                vectorizer = load_model('news_vectorizer.pkl')

                text_vector = vectorizer.transform([user_text])
                pred = model.predict(text_vector)[0]

                if pred == 1:
                    st.error("🚨 WARNING: This news is most likely FAKE (fabricated/manipulative content)!")
                else:
                    st.success("📰 RESULT: This news appears to be REAL and based on verified sources.")
            except FileNotFoundError:
                st.warning("Required model files are missing.")
