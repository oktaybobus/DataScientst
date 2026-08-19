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

    with st.expander(f"📂 Veri Seti Bilgisi: {info['name']}"):
        st.markdown(f"**Kaynak:** {info['source']}")
        st.markdown(f"**Büyüklük:** {info['size']}")
        st.markdown(info["description"])

        if info.get("columns"):
            st.markdown("**Sütun Açıklamaları:**")
            col_df = pd.DataFrame({
                "Sütun": list(info["columns"].keys()),
                "Açıklama": list(info["columns"].values()),
            })
            st.dataframe(col_df, width="stretch", hide_index=True)

        if info.get("sample") is not None:
            st.markdown("**Örnek Veri (ilk birkaç satır):**")
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
    "student":    {"type": "regression",     "R2": 0.881, "RMSE": 5.40, "MAE": 4.10, "unit": "puan"},
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

    "movie_rec": {"type": "none", "note": "İçerik tabanlı (unsupervised) öneri sistemi — accuracy/R² gibi tek bir "
                                            "başarı metriği yoktur. Kalite genelde manuel/precision@k ile değerlendirilir."},
    "book_rec":  {"type": "none", "note": "İçerik tabanlı (unsupervised) öneri sistemi — tek bir başarı metriği yoktur."},
    "song_rec":  {"type": "none", "note": "İçerik tabanlı (unsupervised) öneri sistemi — tek bir başarı metriği yoktur."},

    "text_gen":  {"type": "none", "note": "Markov Zinciri üretken (generative) bir modeldir — "
                                            "doğruluk/hata metriğiyle değil, üretilen metnin akıcılığıyla değerlendirilir."},

    "social_media_viz": {"type": "none", "note": "Bu sekmede eğitilen bir model yok, sadece görselleştirme var."},
    "co2_viz":          {"type": "none", "note": "Bu sekmede eğitilen bir model yok, sadece görselleştirme var."},
    "ecommerce_viz":    {"type": "none", "note": "Bu sekmede eğitilen bir model yok, sadece görselleştirme var."},

    "drowsy":          {"type": "none", "note": "Şu an rastgele simülasyon çalışıyor — ölçülebilir bir başarı puanı yok."},
    "hand":            {"type": "none", "note": "Pretrained MediaPipe modeli kullanılıyor — kendi başarı puanımız yok."},
    "farm_agent":      {"type": "none", "note": "Kural tabanlı ajan — eğitilmiş bir model olmadığı için başarı puanı yok."},
    "faq_agent":       {"type": "none", "note": "Kural tabanlı ajan — eğitilmiş bir model olmadığı için başarı puanı yok."},
    "autonomous_car":  {"type": "none", "note": "Kural tabanlı ajan — eğitilmiş bir model olmadığı için başarı puanı yok."},
}


def show_model_metrics(key: str):
    is_real = key in REAL_METRICS
    m = REAL_METRICS.get(key) or PLACEHOLDER_METRICS.get(key)
    if not m:
        return

    with st.expander("🏆 Model Başarı Puanları"):
        if is_real:
            st.success("✅ Bu puanlar train.py'ın gerçek test-seti değerlendirmesinden geliyor.")
        mtype = m.get("type")

        if mtype == "regression":
            unit = m.get("unit", "")
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Skoru", f"{m['R2']:.3f}")
            c2.metric("RMSE", f"{m['RMSE']:,.2f} {unit}")
            c3.metric("MAE", f"{m['MAE']:,.2f} {unit}")

        elif mtype == "classification":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"%{m['Accuracy']*100:.1f}")
            c2.metric("Precision", f"%{m['Precision']*100:.1f}")
            c3.metric("Recall", f"%{m['Recall']*100:.1f}")
            c4.metric("F1 Skoru", f"%{m['F1']*100:.1f}")

        elif mtype == "clustering":
            c1, c2 = st.columns(2)
            c1.metric("Silhouette Skoru", f"{m['Silhouette']:.2f}")
            c2.metric("Küme Sayısı", m["n_clusters"])
            st.caption("Silhouette Skoru -1 ile 1 arasındadır; 1'e yakın değerler kümelerin "
                       "birbirinden iyi ayrıştığını gösterir.")

        elif mtype == "deep_learning":
            c1, c2, c3 = st.columns(3)
            c1.metric("Eğitim Accuracy", f"%{m['Accuracy']*100:.1f}")
            c2.metric("Validation Accuracy", f"%{m['Val_Accuracy']*100:.1f}")
            c3.metric("Loss", f"{m['Loss']:.3f}")

        elif mtype == "none":
            st.info(m.get("note", "Bu proje için ölçülebilir bir başarı metriği tanımlı değil."))

        if not is_real:
            st.caption("⚠️ Bu puanlar temsilidir (models/metrics.json henüz bulunamadı). "
                       "train.py'ı çalıştırınca gerçek sonuçlar otomatik görünecek.")
