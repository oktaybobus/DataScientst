import streamlit as st
import numpy as np

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", ["Mobile Device Price Segment", "Wine Quality Classification", "Customer Churn Prediction"])

    # ----------------- 1. MOBILE DEVICE PRICE SEGMENT -----------------
    if project == "Mobile Device Price Segment":
        st.header("📱 Mobile Device Price Segment Prediction")
        st.write("Predict which price segment a phone belongs to based on its specifications.")

        show_dataset_info("mobile")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["mobile"], language="python")

        show_model_metrics("mobile")

        col1, col2 = st.columns(2)
        with col1:
            ram = st.slider("RAM Capacity (MB)", 256, 4096, 2048)
            battery = st.slider("Battery Power (mAh)", 500, 5000, 2000)
            int_mem = st.slider("Internal Memory (GB)", 2, 64, 32)
            weight = st.number_input("Phone Weight (Grams)", min_value=80, max_value=250, value=150)
            cores = st.selectbox("Number of Cores (n_cores)", [1, 2, 4, 6, 8])
        with col2:
            clock = st.number_input("Processor Speed (Clock Speed)", min_value=0.5, max_value=3.0, value=1.5)
            m_dep = st.number_input("Phone Thickness (m_dep)", min_value=0.1, max_value=1.0, value=0.5)
            pc = st.slider("Rear Camera Resolution (Megapixel)", 0, 20, 10)
            sim = st.selectbox("Dual SIM?", ["No", "Yes"])
            touch = st.selectbox("Touch Screen?", ["No", "Yes"])
            wifi = st.selectbox("Wi-Fi?", ["No", "Yes"])

        if st.button("Predict Price Segment"):
            model = load_model('mobile_model.pkl')
            sim_val = 1 if sim == "Yes" else 0
            touch_val = 1 if touch == "Yes" else 0
            wifi_val = 1 if wifi == "Yes" else 0

            features = np.array([[battery, clock, sim_val, int_mem, m_dep, weight, cores, pc, ram, touch_val, wifi_val]])
            pred = model.predict(features)[0]

            segments = {0: "📱 Budget Segment", 1: "💵 Mid-Range Segment", 2: "💸 Premium Segment", 3: "🚀 Flagship / Ultra Premium"}
            st.success(f"Predicted Class: **{segments[pred]}**")

    # ----------------- 2. WINE QUALITY CLASSIFICATION -----------------
    elif project == "Wine Quality Classification":
        st.header("🍷 Red Wine Quality Classification")

        show_dataset_info("wine")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["wine"], language="python")

        show_model_metrics("wine")

        col1, col2 = st.columns(2)
        with col1:
            fixed_acid = st.number_input("Fixed Acidity", value=8.3)
            vol_acid = st.number_input("Volatile Acidity", value=0.52)
            citric = st.number_input("Citric Acid", value=0.26)
            sugar = st.number_input("Residual Sugar", value=2.2)
            chlorides = st.number_input("Chlorides", value=0.08)
        with col2:
            free_sulfur = st.number_input("Free Sulfur Dioxide", value=11.0)
            total_sulfur = st.number_input("Total Sulfur Dioxide", value=34.0)
            density = st.number_input("Density", value=0.99)
            ph = st.number_input("pH Value", value=3.31)
            sulphates = st.number_input("Sulphates", value=0.56)
            alcohol = st.number_input("Alcohol (%)", value=10.5)

        if st.button("Rate Wine Quality"):
            model = load_model('wine_model.pkl')
            features = np.array([[fixed_acid, vol_acid, citric, sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]])
            pred = model.predict(features)[0]
            st.success(f"🎯 Predicted Wine Quality Score: **{pred} / 10**")

    # ----------------- 3. CUSTOMER CHURN PREDICTION -----------------
    elif project == "Customer Churn Prediction":
        st.header("🚀 Telecom Customer Churn Analysis")

        show_dataset_info("churn")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["churn"], language="python")

        show_model_metrics("churn")

        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charge ($)", value=65.0)
            total = st.number_input("Total Charges ($)", value=780.0)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            senior = st.selectbox("Senior Citizen (65+)?", ["No", "Yes"])
            partner = st.selectbox("Has Partner?", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            phone = st.selectbox("Phone Service?", ["No", "Yes"])

        if st.button("Calculate Churn Risk"):
            model = load_model('churn_model.pkl')
            g_val = 1 if gender == "Male" else 0
            s_val = 1 if senior == "Yes" else 0
            p_val = 1 if partner == "Yes" else 0
            d_val = 1 if dependents == "Yes" else 0
            ph_val = 1 if phone == "Yes" else 0

            features = np.array([[g_val, s_val, p_val, d_val, tenure, ph_val, monthly, total]])
            pred = model.predict(features)[0]

            if pred == 1:
                st.error("⚠️ Warning: This customer has a HIGH risk of **CHURNING**!")
            else:
                st.success("✅ Good news: This customer is likely to **STAY** with the company.")
