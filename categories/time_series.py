import streamlit as st
import numpy as np
import pandas as pd

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "Stock Price Prediction",
        "Weather / Temperature Prediction",
        "Store Sales Prediction"
    ])

    # ----------------- 1. STOCK PRICE PREDICTION -----------------
    if project == "Stock Price Prediction":
        st.header("📈 Stock Price Prediction (Apple - AAPL)")
        st.write("Simulate the next 7 days of price trends based on Apple stock historical data.")

        show_dataset_info("stock")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["stock"], language="python")

        show_model_metrics("stock")

        days_to_predict = st.slider("Number of Days to Predict", 1, 14, 7)

        if st.button("Predict Future Trend"):
            try:
                model = load_model('stock_model.pkl')
                recent_data = load_model('stock_recent.pkl')  # Son 30 gunun kapanis fiyatlari

                # Gelecek günleri tahmin etme döngüsü
                predictions = []
                current_input = recent_data[-1]

                for _ in range(days_to_predict):
                    pred = model.predict(np.array([[current_input]]))[0]
                    predictions.append(pred)
                    current_input = pred

                # Grafik için veri birleştirme
                history_df = pd.DataFrame({'Price ($)': recent_data, 'Type': 'Historical'})
                future_df = pd.DataFrame({'Price ($)': predictions, 'Type': 'Forecast'})
                total_df = pd.concat([history_df, future_df]).reset_index(drop=True)

                # Grafik Çizimi
                st.write("📊 **Last 30 Days and Forecast Chart**")
                st.line_chart(total_df['Price ($)'])
                st.success(f"🚀 Predicted closing price in {days_to_predict} days: **${predictions[-1]:.2f}**")
            except FileNotFoundError:
                st.warning("Model files are missing. Please run 'train_timeseries.py' first.")

    # ----------------- 2. WEATHER / TEMPERATURE PREDICTION -----------------
    elif project == "Weather / Temperature Prediction":
        st.header("🌦️ Weather / Average Temperature Prediction")
        st.write("Forecast upcoming temperature changes based on historical climate data.")

        show_dataset_info("weather")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["weather"], language="python")

        show_model_metrics("weather")

        days_to_predict = st.slider("Number of Days to Predict", 1, 10, 5, key="weather_days")

        if st.button("Calculate Temperature Trend"):
            try:
                model = load_model('weather_model.pkl')
                recent_data = load_model('weather_recent.pkl')

                predictions = []
                current_input = recent_data[-1]
                for _ in range(days_to_predict):
                    pred = model.predict(np.array([[current_input]]))[0]
                    predictions.append(pred)
                    current_input = pred

                history_df = pd.DataFrame({'Temperature (°C)': recent_data})
                future_df = pd.DataFrame({'Temperature (°C)': predictions})
                total_df = pd.concat([history_df, future_df]).reset_index(drop=True)

                st.line_chart(total_df['Temperature (°C)'])
                st.info(f"🌡️ Predicted average temperature in {days_to_predict} days: **{predictions[-1]:.1f}°C**")
            except FileNotFoundError:
                st.warning("Weather model files are missing.")

    # ----------------- 3. STORE SALES PREDICTION -----------------
    elif project == "Store Sales Prediction":
        st.header("🛒 Store Sales Prediction (Walmart Analytics)")
        st.write("Analyze weekly revenue data to forecast upcoming weeks' sales demand.")

        show_dataset_info("walmart")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["walmart"], language="python")

        show_model_metrics("walmart")

        weeks_to_predict = st.slider("Number of Weeks to Predict", 1, 4, 2)

        if st.button("Predict Weekly Revenue"):
            try:
                model = load_model('walmart_model.pkl')
                recent_data = load_model('walmart_recent.pkl')

                predictions = []
                current_input = recent_data[-1]
                for _ in range(weeks_to_predict):
                    pred = model.predict(np.array([[current_input]]))[0]
                    predictions.append(pred)
                    current_input = pred

                history_df = pd.DataFrame({'Weekly Sales ($)': recent_data})
                future_df = pd.DataFrame({'Weekly Sales ($)': predictions})
                total_df = pd.concat([history_df, future_df]).reset_index(drop=True)

                st.line_chart(total_df['Weekly Sales ($)'])
                st.success(f"💳 Predicted store revenue in {weeks_to_predict} weeks: **${predictions[-1]:,.2f}**")
            except FileNotFoundError:
                st.warning("Walmart model files not found.")
