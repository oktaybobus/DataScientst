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

st.set_page_config(page_title="Machine Learning Portfolio", layout="wide")
st.title("Specialize in Data Science")

main_category = st.sidebar.selectbox("Select Category", [
    "Regression",
    "Classification",
    "Clustering",
    "Computer Vision",
    "Natural Language Processing (NLP)",
    "Recommendation Systems",
    "Time Series",
    "Data Visualization",
    "Deep Learning",
    "AI Agents"
])

if main_category == "Regression":
    regression.render()
elif main_category == "Classification":
    classification.render()
elif main_category == "Clustering":
    clustering.render()
elif main_category == "Computer Vision":
    computer_vision.render()
elif main_category == "Natural Language Processing (NLP)":
    nlp.render()
elif main_category == "Recommendation Systems":
    recommendation.render()
elif main_category == "Time Series":
    time_series.render()
elif main_category == "Data Visualization":
    data_viz.render()
elif main_category == "Deep Learning":
    deep_learning.render()
elif main_category == "AI Agents":
    ai_agents.render()
