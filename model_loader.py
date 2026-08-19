"""
HF Hub'dan model indirme yardimcisi.
Yerel 'models/' klasoru varsa oradan yukler, yoksa HF'den indirir ve onbellekler.
"""
import os
import joblib

HF_REPO = "OKTAYBBS/DataScientst-models"

# Kategori eslesmesi: dosya adi -> HF klasoru
_CATEGORY_MAP = {
    "gold_model.pkl": "regression",
    "student_model.pkl": "regression",
    "uber_model.pkl": "regression",
    "mobile_model.pkl": "classification",
    "wine_model.pkl": "classification",
    "churn_model.pkl": "classification",
    "nba_model.pkl": "clustering",
    "nba_scaler.pkl": "clustering",
    "cc_model.pkl": "clustering",
    "cc_scaler.pkl": "clustering",
    "spotify_model.pkl": "clustering",
    "spotify_scaler.pkl": "clustering",
    "mask_model.pkl": "computer_vision",
    "spam_model.pkl": "nlp",
    "spam_vectorizer.pkl": "nlp",
    "imdb_model.pkl": "nlp",
    "imdb_vectorizer.pkl": "nlp",
    "news_model.pkl": "nlp",
    "news_vectorizer.pkl": "nlp",
    "movie_data.pkl": "recommendation",
    "movie_similarity.pkl": "recommendation",
    "book_data.pkl": "recommendation",
    "book_similarity.pkl": "recommendation",
    "song_data.pkl": "recommendation",
    "song_similarity.pkl": "recommendation",
    "stock_model.pkl": "time_series",
    "stock_recent.pkl": "time_series",
    "weather_model.pkl": "time_series",
    "weather_recent.pkl": "time_series",
    "walmart_model.pkl": "time_series",
    "walmart_recent.pkl": "time_series",
    "social_media_viz.csv": "data_viz",
    "co2_emissions_viz.csv": "data_viz",
    "ecommerce_viz.csv": "data_viz",
    "pneumonia_model.keras": "deep_learning",
    "fer_model.keras": "deep_learning",
    "text_robot_model.pkl": "deep_learning",
    "metrics.json": "metrics",
    "metrics_summary.csv": "metrics",
    "metrics_summary.png": "metrics",
}


def get_model_path(filename):
    """
    Yerel models/ klasorunde varsa oradan dondurur,
    yoksa HF Hub'dan indirir ve cache'lenmiş yolu dondurur.
    """
    local_path = os.path.join("models", filename)
    if os.path.exists(local_path):
        return local_path

    from huggingface_hub import hf_hub_download
    category = _CATEGORY_MAP.get(filename, "")
    repo_path = f"{category}/{filename}" if category else filename
    return hf_hub_download(repo_id=HF_REPO, filename=repo_path)


def load_model(filename):
    """joblib.load ile model yukler (yerel veya HF)."""
    return joblib.load(get_model_path(filename))


def load_keras_model(filename):
    """TensorFlow/Keras model yukler (yerel veya HF)."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow bu ortamda yuklu degil. "
            "Derin ogrenme modelleri icin: pip install tensorflow"
        )
    return tf.keras.models.load_model(get_model_path(filename))
