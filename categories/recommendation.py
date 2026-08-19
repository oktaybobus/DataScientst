import streamlit as st

from model_loader import load_model
from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select a Project", [
        "Movie Recommendation System",
        "Book Recommendation Engine",
        "Music Recommendation System"
    ])

    # ----------------- 1. MOVIE RECOMMENDATION SYSTEM -----------------
    if project == "Movie Recommendation System":
        st.header("🍿 Movie Recommendation System")
        st.write("Select a movie you watched and enjoyed, and we will recommend similar movies.")

        show_dataset_info("movie_rec")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["movie_rec"], language="python")

        show_model_metrics("movie_rec")

        try:
            movies_df = load_model('movie_data.pkl')
            movie_sim = load_model('movie_similarity.pkl')

            movie_list = movies_df['title'].values
            selected_movie = st.selectbox("Select a Movie", movie_list)

            if st.button("Recommend Similar Movies"):
                # Seçilen filmin indeksini bulma
                idx = movies_df[movies_df['title'] == selected_movie].index[0]
                # Benzerlik skorlarını sıralama (kendisi hariç)
                sim_scores = list(enumerate(movie_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.subheader("🎬 Movies Selected for You:")
                for i, score in enumerate(sim_scores):
                    st.write(f"{i+1}. **{movies_df.iloc[score[0]]['title']}** (Similarity Score: %{score[1]*100:.1f})")
        except FileNotFoundError:
            st.warning("Model files not found. Please run 'train_recommender.py' first.")

    # ----------------- 2. BOOK RECOMMENDATION ENGINE -----------------
    elif project == "Book Recommendation Engine":
        st.header("📚 Book Recommendation Engine")
        st.write("Select a book you have read, and discover new books based on author and content similarity.")

        show_dataset_info("book_rec")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["book_rec"], language="python")

        show_model_metrics("book_rec")

        try:
            books_df = load_model('book_data.pkl')
            book_sim = load_model('book_similarity.pkl')

            book_list = books_df['title'].values
            selected_book = st.selectbox("Select a Book", book_list)

            if st.button("Recommend Similar Books"):
                idx = books_df[books_df['title'] == selected_book].index[0]
                sim_scores = list(enumerate(book_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.subheader("📖 Books to Add to Your Reading List:")
                for i, score in enumerate(sim_scores):
                    book_title = books_df.iloc[score[0]]['title']
                    book_author = books_df.iloc[score[0]]['authors']
                    st.write(f"{i+1}. **{book_title}** - *Author: {book_author}*")
        except FileNotFoundError:
            st.warning("Book model files are missing.")

    # ----------------- 3. MUSIC RECOMMENDATION SYSTEM -----------------
    elif project == "Music Recommendation System":
        st.header("🎵 Music Recommendation System")
        st.write("Select a song that matches your mood, and we will find tracks with similar rhythm and genre.")

        show_dataset_info("song_rec")

        with st.expander("🧪 How Was the Model Trained? (View Training Code)"):
            st.code(TRAIN_CODE["song_rec"], language="python")

        show_model_metrics("song_rec")

        try:
            songs_df = load_model('song_data.pkl')
            song_sim = load_model('song_similarity.pkl')

            song_list = songs_df['track_name'].values
            selected_song = st.selectbox("Select a Song", song_list)

            if st.button("Recommend Similar Songs"):
                idx = songs_df[songs_df['track_name'] == selected_song].index[0]
                sim_scores = list(enumerate(song_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.subheader("🎧 Up Next (Queue):")
                for i, score in enumerate(sim_scores):
                    track_name = songs_df.iloc[score[0]]['track_name']
                    artist_name = songs_df.iloc[score[0]]['artists']
                    st.write(f"{i+1}. **{track_name}** - *Artist: {artist_name}*")
        except FileNotFoundError:
            st.warning("Song model files are missing.")
