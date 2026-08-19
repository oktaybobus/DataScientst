import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import streamlit as st

from categories import regression
from categories import classification
from categories import clustering
from categories import computer_vision
from categories import nlp
from categories import recommendation
from categories import time_series
from categories import data_viz
from categories import deep_learning
from categories import ai_agents

st.set_page_config(page_title="Makine Öğrenmesi Portfolyosu", layout="wide")
st.title("Specialize in Data Science")

# Sol Menü Güncellemesi (Tüm 10 Kategori Tamamlandı!)
main_category = st.sidebar.selectbox("Uygulama Kategorisi Seçin", [
    "Regresyon (Regression)",
    "Sınıflandırma (Classification)",
    "Kümeleme (Clustering)",
    "Bilgisayarlı Görü (Computer Vision)",
    "Doğal Dil İşleme (NLP)",
    "Öneri Sistemleri (Recommendation)",
    "Zaman Serileri (Time Series)",
    "Veri Görselleştirme (Data Viz)",
    "Derin Öğrenme (Deep Learning)",
    "Yapay Zeka Ajanları (AI Agents)"
])

if main_category == "Regresyon (Regression)":
    regression.render()
elif main_category == "Sınıflandırma (Classification)":
    classification.render()
elif main_category == "Kümeleme (Clustering)":
    clustering.render()
elif main_category == "Bilgisayarlı Görü (Computer Vision)":
    computer_vision.render()
elif main_category == "Doğal Dil İşleme (NLP)":
    nlp.render()
elif main_category == "Öneri Sistemleri (Recommendation)":
    recommendation.render()
elif main_category == "Zaman Serileri (Time Series)":
    time_series.render()
elif main_category == "Veri Görselleştirme (Data Viz)":
    data_viz.render()
elif main_category == "Derin Öğrenme (Deep Learning)":
    deep_learning.render()
elif main_category == "Yapay Zeka Ajanları (AI Agents)":
    ai_agents.render()
