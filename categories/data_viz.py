import streamlit as st
import pandas as pd
import plotly.express as px

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "Küresel Sosyal Medya İstatistikleri",
        "İklim Değişikliği & Karbon Salınımı",
        "E-Ticaret Demografi Dashboard"
    ])

    # ----------------- 1. KÜRESEL SOSYAL MEDYA İSTATİSTİKLERİ -----------------
    if project == "Küresel Sosyal Medya İstatistikleri":
        st.header("📱 Küresel Sosyal Medya Kullanım İstatistikleri")
        st.write("Farklı yaş grupları ve cinsiyetlerin sosyal medya kullanım alışkanlıklarını inceleyin.")

        show_dataset_info("social_media_viz")

        with st.expander("📂 Veri Hazırlama Kodunu Gör"):
            st.code(TRAIN_CODE["social_media_viz"], language="python")

        show_model_metrics("social_media_viz")

        try:
            df = pd.read_csv('models/social_media_viz.csv')

            # Dinamik Filtreler
            gender_filter = st.multiselect("Cinsiyet Seçin", df['Gender'].unique(), default=df['Gender'].unique())
            filtered_df = df[df['Gender'].isin(gender_filter)]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Platformlara Göre Günlük Ortalama Süre (Dakika)")
                fig1 = px.box(filtered_df, x='Platform', y='Daily_Usage_Time (minutes)', color='Platform', points="all")
                st.plotly_chart(fig1, width="stretch")

            with col2:
                st.subheader("🎭 Yaş Gruplarına Göre Baskın Duygu Durumu")
                fig2 = px.histogram(filtered_df, x='Age', color='Dominant_Emotion', barmode='group')
                st.plotly_chart(fig2, width="stretch")

        except FileNotFoundError:
            st.warning("Veri dosyası bulunamadı. Lütfen önce 'prepare_viz_data.py' dosyasını çalıştırın.")

    # ----------------- 2. İKLİM DEĞİŞİKLİĞİ & KARBON SALINIMI -----------------
    elif project == "İklim Değişikliği & Karbon Salınımı":
        st.header("⚡ Küresel Karbon Salınımı ve İklim Değişikliği Paneli")
        st.write("Ülkelerin yıllara göre fosil yakıt kaynaklı CO2 emisyon trendlerini yarıştırın.")

        show_dataset_info("co2_viz")

        with st.expander("📂 Veri Hazırlama Kodunu Gör"):
            st.code(TRAIN_CODE["co2_viz"], language="python")

        show_model_metrics("co2_viz")

        try:
            df = pd.read_csv('models/co2_emissions_viz.csv')

            # Ülke Seçimi
            country_list = df['Country'].unique()
            selected_countries = st.multiselect("Karşılaştırılacak Ülkeleri Seçin", country_list, default=["United States", "China", "United Kingdom"])

            filtered_df = df[df['Country'].isin(selected_countries)]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Yıllara Göre Toplam CO2 Salınımı Trendi")
                fig1 = px.line(filtered_df, x='Year', y='Total', color='Country', markers=True)
                st.plotly_chart(fig1, width="stretch")

            with col2:
                st.subheader("🪨 Emisyon Kaynaklarının Dağılımı (Kömür, Petrol, Gaz)")
                # Son yıla göre kaynak dağılımı bar grafiği
                latest_year = filtered_df['Year'].max()
                latest_df = filtered_df[filtered_df['Year'] == latest_year]
                fig2 = px.bar(latest_df, x='Country', y=['Coal', 'Oil', 'Gas'], title=f"{latest_year} Yılı Detayı", barmode='stack')
                st.plotly_chart(fig2, width="stretch")

        except FileNotFoundError:
            st.warning("CO2 emisyon veri dosyası eksik.")

    # ----------------- 3. E-TİCARET DEMOGRAFİ DASHBOARD -----------------
    elif project == "E-Ticaret Demografi Dashboard":
        st.header("🛍️ E-Ticaret Satış ve Müşteri Demografisi Paneli")
        st.write("Online mağazanın satış performansını, ürün popülerliğini ve ülke dağılımlarını analiz edin.")

        show_dataset_info("ecommerce_viz")

        with st.expander("📂 Veri Hazırlama Kodunu Gör"):
            st.code(TRAIN_CODE["ecommerce_viz"], language="python")

        show_model_metrics("ecommerce_viz")

        try:
            df = pd.read_csv('models/ecommerce_viz.csv')

            # Toplam İstatistikler (KPI Cards)
            total_revenue = df['Total_Price'].sum()
            total_sales = df['Quantity'].sum()

            c1, c2 = st.columns(2)
            c1.metric("💰 Toplam Ciro", f"${total_revenue:,.2f}")
            c2.metric("📦 Satılan Toplam Ürün Adedi", f"{total_sales:,}")

            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🌍 Ülkelere Göre Ciro Dağılımı")
                country_rev = df.groupby('Country')['Total_Price'].sum().reset_index()
                fig1 = px.pie(country_rev, values='Total_Price', names='Country', hole=0.4)
                st.plotly_chart(fig1, width="stretch")

            with col2:
                st.subheader("🔝 En Çok Satan İlk 10 Ürün (Açıklama)")
                top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
                fig2 = px.bar(top_products, x='Quantity', y='Description', orientation='h', color='Quantity')
                st.plotly_chart(fig2, width="stretch")

        except FileNotFoundError:
            st.warning("E-ticaret veri dosyası eksik.")
