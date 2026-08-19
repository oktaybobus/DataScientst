import streamlit as st
import pandas as pd
import json

from model_loader import get_model_path
from dataset_info import DATASET_INFO


# ====================================================================
# Gosterim Fonksiyonu -- her proje blogunda cagir
# Kullanim: show_dataset_info("gold")
# ====================================================================
def show_dataset_info(key: str):
    info = DATASET_INFO.get(key)
    if not info:
        return

    with st.expander(f"📂 Dataset Info: {info['name']}"):
        st.markdown(f"**Source:** {info['source']}")
        st.markdown(f"**Size:** {info['size']}")
        st.markdown(info["description"])

        if info.get("columns"):
            st.markdown("**Column Descriptions:**")
            col_df = pd.DataFrame({
                "Column": list(info["columns"].keys()),
                "Description": list(info["columns"].values()),
            })
            st.dataframe(col_df, width="stretch", hide_index=True)

        if info.get("sample") is not None:
            st.markdown("**Sample Data (first rows):**")
            st.dataframe(info["sample"], width="stretch")

        if info.get("note"):
            st.info(info["note"])


# ====================================================================
# GERCEK METRIKLER: train.py calistirildiginda models/metrics.json
# dosyasina yazilir. Bu dosya varsa GERCEK sayilar kullanilir.
# ====================================================================

try:
    METRICS_PATH = get_model_path("metrics.json")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        REAL_METRICS = json.load(f)
except (FileNotFoundError, json.JSONDecodeError, Exception):
    REAL_METRICS = {}

# ====================================================================
# PLACEHOLDER_METRICS: models/metrics.json henuz uretilmediyse
# (yani train.py hic calistirilmadiysa) gosterilecek temsili yedek
# degerler. train.py'i calistirip gercek metrics.json olusunca bu
# sozluk artik KULLANILMAZ -- otomatik olarak gercek degerlere gecilir.
# ====================================================================

PLACEHOLDER_METRICS = {
    "gold":       {"type": "regression",     "R2": 0.984, "RMSE": 2.14, "MAE": 1.52, "unit": "$"},
    "student":    {"type": "regression",     "R2": 0.881, "RMSE": 5.40, "MAE": 4.10, "unit": "pts"},
    "uber":       {"type": "regression",     "R2": 0.724, "RMSE": 3.80, "MAE": 2.10, "unit": "$"},

    "mobile":     {"type": "classification", "Accuracy": 0.892, "Precision": 0.885, "Recall": 0.890, "F1": 0.887},
    "wine":       {"type": "classification", "Accuracy": 0.681, "Precision": 0.652, "Recall": 0.681, "F1": 0.660},
    "churn":      {"type": "classification", "Accuracy": 0.803, "Precision": 0.663, "Recall": 0.551, "F1": 0.602},
    "mask":       {"type": "classification", "Accuracy": 0.941, "Precision": 0.935, "Recall": 0.947, "F1": 0.941},
    "sms":        {"type": "classification", "Accuracy": 0.972, "Precision": 0.961, "Recall": 0.928, "F1": 0.944},
    "imdb_sentiment": {"type": "classification", "Accuracy": 0.891, "Precision": 0.889, "Recall": 0.884, "F1": 0.886},
    "fake_news":  {"type": "classification", "Accuracy": 0.941, "Precision": 0.934, "Recall": 0.949, "F1": 0.941},

    "nba":        {"type": "clustering", "Silhouette": 0.42, "n_clusters": 4},
    "creditcard": {"type": "clustering", "Silhouette": 0.51, "n_clusters": 4},
    "spotify":    {"type": "clustering", "Silhouette": 0.39, "n_clusters": 4},

    "stock":      {"type": "regression", "R2": 0.912, "RMSE": 3.20, "MAE": 2.40, "unit": "$"},
    "weather":    {"type": "regression", "R2": 0.854, "RMSE": 1.80, "MAE": 1.30, "unit": "°C"},
    "walmart":    {"type": "regression", "R2": 0.831, "RMSE": 45210, "MAE": 32150, "unit": "$"},

    "pneumonia":    {"type": "deep_learning", "Accuracy": 0.912, "Val_Accuracy": 0.887, "Loss": 0.281},
    "face_emotion": {"type": "deep_learning", "Accuracy": 0.641, "Val_Accuracy": 0.598, "Loss": 0.953},

    "movie_rec": {"type": "none", "note": "Content-based (unsupervised) recommendation system — there is no single "
                                            "accuracy/R² metric. Quality is typically evaluated manually or via precision@k."},
    "book_rec":  {"type": "none", "note": "Content-based (unsupervised) recommendation system — there is no single accuracy metric."},
    "song_rec":  {"type": "none", "note": "Content-based (unsupervised) recommendation system — there is no single accuracy metric."},

    "text_gen":  {"type": "none", "note": "A Markov Chain is a generative model — "
                                            "it is evaluated by text fluency, not accuracy/error metrics."},

    "social_media_viz": {"type": "none", "note": "No trained model in this tab — visualization only."},
    "co2_viz":          {"type": "none", "note": "No trained model in this tab — visualization only."},
    "ecommerce_viz":    {"type": "none", "note": "No trained model in this tab — visualization only."},

    "drowsy":          {"type": "none", "note": "Currently running a random simulation — no measurable accuracy score."},
    "hand":            {"type": "none", "note": "Uses a pretrained MediaPipe model — no custom accuracy score available."},
    "farm_agent":      {"type": "none", "note": "Rule-based agent — no trained model, so no accuracy score."},
    "faq_agent":       {"type": "none", "note": "Rule-based agent — no trained model, so no accuracy score."},
    "autonomous_car":  {"type": "none", "note": "Rule-based agent — no trained model, so no accuracy score."},
}


def show_model_metrics(key: str):
    is_real = key in REAL_METRICS
    m = REAL_METRICS.get(key) or PLACEHOLDER_METRICS.get(key)
    if not m:
        return

    with st.expander("🏆 Model Performance Metrics"):
        if is_real:
            st.success("✅ These scores come from actual test-set evaluation.")
        mtype = m.get("type")

        if mtype == "regression":
            unit = m.get("unit", "")
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{m['R2']:.3f}")
            c2.metric("RMSE", f"{m['RMSE']:,.2f} {unit}")
            c3.metric("MAE", f"{m['MAE']:,.2f} {unit}")

        elif mtype == "classification":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"%{m['Accuracy']*100:.1f}")
            c2.metric("Precision", f"%{m['Precision']*100:.1f}")
            c3.metric("Recall", f"%{m['Recall']*100:.1f}")
            c4.metric("F1 Score", f"%{m['F1']*100:.1f}")

        elif mtype == "clustering":
            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score", f"{m['Silhouette']:.2f}")
            c2.metric("Number of Clusters", m["n_clusters"])
            st.caption("Silhouette Score ranges from -1 to 1; values closer to 1 indicate "
                       "well-separated clusters.")

        elif mtype == "deep_learning":
            c1, c2, c3 = st.columns(3)
            c1.metric("Training Accuracy", f"%{m['Accuracy']*100:.1f}")
            c2.metric("Validation Accuracy", f"%{m['Val_Accuracy']*100:.1f}")
            c3.metric("Loss", f"{m['Loss']:.3f}")

        elif mtype == "none":
            st.info(m.get("note", "No measurable accuracy metric is defined for this project."))

        if not is_real:
            st.caption("⚠️ These are placeholder values (models/metrics.json not found yet). "
                       "Run train.py and the real scores will appear automatically.")
