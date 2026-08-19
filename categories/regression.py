import streamlit as st
import numpy as np
import pandas as pd

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", ["Gold Price Prediction", "Student Exam Score Prediction", "Uber/Taxi Fare Prediction"])

    if project == "Gold Price Prediction":
        st.header("🪙 Gold Price Prediction Using Global Indices")
        st.info(
            "This model predicts the gold ounce price using global financial indicators such as "
            "the S&P 500 index, crude oil price (USO), silver price (SLV), and EUR/USD exchange rate."
        )

        show_dataset_info("gold")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["gold"], language="python")

        show_model_metrics("gold")


        col1, col2 = st.columns(2)
        with col1:
            spx = st.number_input("S&P 500 Index (SPX)", value=2200.0)
            uso = st.number_input("Crude Oil Price (USO)", value=40.0)
        with col2:
            slv = st.number_input("Silver Price (SLV)", value=15.0)
            eur_usd = st.number_input("EUR/USD Exchange Rate", value=1.12)
        if st.button("Predict Gold Price"):
            model = load_model('gold_model.pkl')
            pred = model.predict(np.array([[spx, uso, slv, eur_usd]]))
            st.success(f"💰 Predicted Gold Ounce Price: **${pred.item():,.2f}**")


# ----------------- 2. STUDENT EXAM SCORE PREDICTION -----------------
    elif project == "Student Exam Score Prediction":
        st.header("📝 Student Math Exam Score Prediction")
        st.write("Predict a student's math score based on their social background and other exam scores.")

        show_dataset_info("student")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["student"], language="python")

        show_model_metrics("student")

        col1, col2 = st.columns(2)
        with col1:
            reading = st.slider("Reading Score", 0, 100, 70)
            writing = st.slider("Writing Score", 0, 100, 70)
            gender = st.selectbox("Gender", ["Female", "Male"])
            prep = st.selectbox("Test Preparation Course Completed?", ["No", "Yes"])
        with col2:
            lunch = st.selectbox("Lunch Type", ["Standard", "Free/Reduced"])
            race = st.selectbox("Race / Ethnicity Group", ["Group A", "Group B", "Group C", "Group D", "Group E"])
            edu = st.selectbox("Parental Education Level", [
                "some high school", "high school", "some college",
                "associate's degree", "bachelor's degree", "master's degree"
            ])

        if st.button("Predict Math Score"):
            try:
                model = load_model('student_model.pkl')

                # One-hot encoding eğitim sütun sırasına göre girdileri True/False (1/0) olarak hazırlama
                g_male = 1 if gender == "Male" else 0
                r_B = 1 if race == "Group B" else 0
                r_C = 1 if race == "Group C" else 0
                r_D = 1 if race == "Group D" else 0
                r_E = 1 if race == "Group E" else 0

                e_asc = 1 if edu == "associate's degree" else 0
                e_bach = 1 if edu == "bachelor's degree" else 0
                e_hs = 1 if edu == "high school" else 0
                e_mast = 1 if edu == "master's degree" else 0
                e_sc = 1 if edu == "some college" else 0

                l_std = 1 if lunch == "Standard" else 0
                p_none = 1 if prep == "No" else 0

                # Model eğitimindeki tam sütun dizilimi:
                features = np.array([[
                    reading, writing, g_male, r_B, r_C, r_D, r_E,
                    e_asc, e_bach, e_hs, e_mast, e_sc, l_std, p_none
                ]])

                pred = model.predict(features)
                st.success(f"🎯 Predicted Math Exam Score: **{int(pred[0])} / 100**")
            except FileNotFoundError:
                st.error("Error: 'models/student_model.pkl' not found. Please train the model first.")

    # ----------------- 3. UBER/TAXI FARE PREDICTION -----------------
    elif project == "Uber/Taxi Fare Prediction":
        st.header("🚗 Uber/Taxi Ride Fare Prediction")
        st.write("Estimate the ride cost based on pickup and dropoff coordinates and passenger count.")

        show_dataset_info("uber")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["uber"], language="python")

        show_model_metrics("uber")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Route Coordinates")
            p_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.4f")
            p_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.4f")
            d_lat = st.number_input("Dropoff Latitude", value=40.7829, format="%.4f")
            d_lon = st.number_input("Dropoff Longitude", value=-73.9654, format="%.4f")
            passengers = st.slider("Passenger Count", 1, 6, 1)

        with col2:
            st.subheader("🗺️ Live Route View")
            # Haritada noktaları göstermek için veri çerçevesi (DataFrame) oluşturuyoruz
            map_data = pd.DataFrame({
                'lat': [p_lat, d_lat],
                'lon': [p_lon, d_lon]
            })
            st.map(map_data)

        if st.button("Calculate Ride Fare"):
            try:
                model = load_model('uber_model.pkl')

                # Model eğitimindeki tam sütun dizilimi:
                # pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count
                features = np.array([[p_lon, p_lat, d_lon, d_lat, passengers]])

                pred = model.predict(features)
                st.success(f"💳 Estimated Ride Fare: **${pred[0]:,.2f}**")
            except FileNotFoundError:
                st.error("Error: 'models/uber_model.pkl' not found. Please train the model first.")
