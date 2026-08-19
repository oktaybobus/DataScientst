import streamlit as st
import joblib

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Bir Proje Seçin", [
        "IMDb Film Öneri Sistemi",
        "Kitap Tavsiye Motoru",
        "Şarkı / Müzik Öneri Sistemi"
    ])

    # ----------------- 1. IMDb FİLM ÖNERİ SİSTEMİ -----------------
    if project == "IMDb Film Öneri Sistemi":
        st.header("🍿 IMDb Film Öneri Sistemi")
        st.write("İzlediğiniz ve beğendiğiniz bir filmi seçin, size benzer tarzdaki diğer filmleri önerelim.")

        show_dataset_info("movie_rec")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["movie_rec"], language="python")

        show_model_metrics("movie_rec")

        try:
            movies_df = joblib.load('models/movie_data.pkl')
            movie_sim = joblib.load('models/movie_similarity.pkl')

            movie_list = movies_df['title'].values
            selected_movie = st.selectbox("Bir Film Seçin", movie_list)

            if st.button("Benzer Filmleri Öner"):
                # Seçilen filmin indeksini bulma
                idx = movies_df[movies_df['title'] == selected_movie].index[0]
                # Benzerlik skorlarını sıralama (kendisi hariç)
                sim_scores = list(enumerate(movie_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.subheader("🎬 Sizin İçin Seçtiğimiz Filmler:")
                for i, score in enumerate(sim_scores):
                    st.write(f"{i+1}. **{movies_df.iloc[score[0]]['title']}** (Benzerlik Skoru: %{score[1]*100:.1f})")
        except FileNotFoundError:
            st.warning("Model dosyaları bulunamadı. Lütfen önce 'train_recommender.py' dosyasını çalıştırın.")

    # ----------------- 2. KİTAP TAVSİYE MOTORU -----------------
    elif project == "Kitap Tavsiye Motoru":
        st.header("📚 Kitap Tavsiye Motoru")
        st.write("Okuduğunuz bir kitabı seçin, yazar ve içerik benzerliğine göre yeni kitap keşfedin.")

        show_dataset_info("book_rec")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["book_rec"], language="python")

        show_model_metrics("book_rec")

        try:
            books_df = joblib.load('models/book_data.pkl')
            book_sim = joblib.load('models/book_similarity.pkl')

            book_list = books_df['title'].values
            selected_book = st.selectbox("Bir Kitap Seçin", book_list)

            if st.button("Benzer Kitapları Öner"):
                idx = books_df[books_df['title'] == selected_book].index[0]
                sim_scores = list(enumerate(book_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.subheader("📖 Okuma Listenize Eklenebilecek Kitaplar:")
                for i, score in enumerate(sim_scores):
                    book_title = books_df.iloc[score[0]]['title']
                    book_author = books_df.iloc[score[0]]['authors']
                    st.write(f"{i+1}. **{book_title}** - *Yazar: {book_author}*")
        except FileNotFoundError:
            st.warning("Kitap model dosyaları eksik.")

    # ----------------- 3. ŞARKI / MÜZİK ÖNERİ SİSTEMİ -----------------
    elif project == "Şarkı / Müzik Öneri Sistemi":
        st.header("🎵 Şarkı / Müzik Öneri Sistemi")
        st.write("Ruh halinize uyan bir şarkı seçin, Spotify tarzı benzer ritim ve türdeki parçaları getirelim.")

        show_dataset_info("song_rec")

        with st.expander("🧪 Model Nasıl Eğitildi? (Eğitim Kodunu Gör)"):
            st.code(TRAIN_CODE["song_rec"], language="python")

        show_model_metrics("song_rec")

        try:
            songs_df = joblib.load('models/song_data.pkl')
            song_sim = joblib.load('models/song_similarity.pkl')

            song_list = songs_df['track_name'].values
            selected_song = st.selectbox("Bir Şarkı Seçin", song_list)

            if st.button("Benzer Şarkıları Öner"):
                idx = songs_df[songs_df['track_name'] == selected_song].index[0]
                sim_scores = list(enumerate(song_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.subheader("🎧 Sıradaki Parçalarınız (Queue):")
                for i, score in enumerate(sim_scores):
                    track_name = songs_df.iloc[score[0]]['track_name']
                    artist_name = songs_df.iloc[score[0]]['artists']
                    st.write(f"{i+1}. **{track_name}** - *Sanatçı: {artist_name}*")
        except FileNotFoundError:
            st.warning("Şarkı model dosyaları eksik.")
