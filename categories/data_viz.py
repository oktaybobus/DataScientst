import streamlit as st
import pandas as pd
import plotly.express as px

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "Global Social Media Statistics",
        "Climate Change & CO2 Emissions",
        "E-Commerce Sales Dashboard"
    ])

    # ----------------- 1. GLOBAL SOCIAL MEDIA STATISTICS -----------------
    if project == "Global Social Media Statistics":
        st.header("📱 Global Social Media Usage Statistics")
        st.write("Explore social media usage habits across different age groups and genders.")

        show_dataset_info("social_media_viz")

        with st.expander("📂 View Data Preparation Code"):
            st.code(TRAIN_CODE["social_media_viz"], language="python")

        show_model_metrics("social_media_viz")

        try:
            df = pd.read_csv('models/social_media_viz.csv')

            # Dinamik Filtreler
            gender_filter = st.multiselect("Select Gender", df['Gender'].unique(), default=df['Gender'].unique())
            filtered_df = df[df['Gender'].isin(gender_filter)]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Average Daily Usage Time by Platform (Minutes)")
                fig1 = px.box(filtered_df, x='Platform', y='Daily_Usage_Time (minutes)', color='Platform', points="all")
                st.plotly_chart(fig1, width="stretch")

            with col2:
                st.subheader("🎭 Dominant Emotion by Age Group")
                fig2 = px.histogram(filtered_df, x='Age', color='Dominant_Emotion', barmode='group')
                st.plotly_chart(fig2, width="stretch")

        except FileNotFoundError:
            st.warning("Data file not found. Please run 'prepare_viz_data.py' first.")

    # ----------------- 2. CLIMATE CHANGE & CO2 EMISSIONS -----------------
    elif project == "Climate Change & CO2 Emissions":
        st.header("⚡ Global CO2 Emissions and Climate Change Panel")
        st.write("Compare countries' fossil fuel CO2 emission trends over the years.")

        show_dataset_info("co2_viz")

        with st.expander("📂 View Data Preparation Code"):
            st.code(TRAIN_CODE["co2_viz"], language="python")

        show_model_metrics("co2_viz")

        try:
            df = pd.read_csv('models/co2_emissions_viz.csv')

            # Ülke Seçimi
            country_list = df['Country'].unique()
            selected_countries = st.multiselect("Select Countries to Compare", country_list, default=["United States", "China", "United Kingdom"])

            filtered_df = df[df['Country'].isin(selected_countries)]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Total CO2 Emissions Trend by Year")
                fig1 = px.line(filtered_df, x='Year', y='Total', color='Country', markers=True)
                st.plotly_chart(fig1, width="stretch")

            with col2:
                st.subheader("🪨 Emission Source Distribution (Coal, Oil, Gas)")
                # Son yıla göre kaynak dağılımı bar grafiği
                latest_year = filtered_df['Year'].max()
                latest_df = filtered_df[filtered_df['Year'] == latest_year]
                fig2 = px.bar(latest_df, x='Country', y=['Coal', 'Oil', 'Gas'], title=f"{latest_year} Details", barmode='stack')
                st.plotly_chart(fig2, width="stretch")

        except FileNotFoundError:
            st.warning("CO2 emissions data file is missing.")

    # ----------------- 3. E-COMMERCE SALES DASHBOARD -----------------
    elif project == "E-Commerce Sales Dashboard":
        st.header("🛍️ E-Commerce Sales and Customer Demographics Panel")
        st.write("Analyze the online store's sales performance, product popularity, and country distribution.")

        show_dataset_info("ecommerce_viz")

        with st.expander("📂 View Data Preparation Code"):
            st.code(TRAIN_CODE["ecommerce_viz"], language="python")

        show_model_metrics("ecommerce_viz")

        try:
            df = pd.read_csv('models/ecommerce_viz.csv')

            # Toplam İstatistikler (KPI Cards)
            total_revenue = df['Total_Price'].sum()
            total_sales = df['Quantity'].sum()

            c1, c2 = st.columns(2)
            c1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
            c2.metric("📦 Total Units Sold", f"{total_sales:,}")

            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🌍 Revenue Distribution by Country")
                country_rev = df.groupby('Country')['Total_Price'].sum().reset_index()
                fig1 = px.pie(country_rev, values='Total_Price', names='Country', hole=0.4)
                st.plotly_chart(fig1, width="stretch")

            with col2:
                st.subheader("🔝 Top 10 Best-Selling Products")
                top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
                fig2 = px.bar(top_products, x='Quantity', y='Description', orientation='h', color='Quantity')
                st.plotly_chart(fig2, width="stretch")

        except FileNotFoundError:
            st.warning("E-commerce data file is missing.")
