import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (
    PROCESSED_DATA,
    CONTENT_OUT,
    OUTPUT_DIR
)


def content_based_recommendations(user_id: int = 1, top_n: int = 10):
    """
    Generate content-based movie recommendations using
    TF-IDF on genres and cosine similarity.
    """

    # ---------- Ensure output directory exists ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load processed data ----------
    df = pd.read_csv(PROCESSED_DATA)

    # ---------- Create movie-level dataframe ----------
    movies_df = (
        df[["movie_id", "title", "genres"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------- TF-IDF Vectorization on genres ----------
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])

    # ---------- Compute cosine similarity ----------
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # ---------- Movies watched by the user ----------
    user_movies = df[df["user_id"] == user_id]["movie_id"].unique()

    if len(user_movies) == 0:
        raise ValueError(f"User {user_id} has no watched movies for content-based filtering.")

    # Map movie_id to index
    movie_id_to_index = {
        movie_id: idx for idx, movie_id in enumerate(movies_df["movie_id"])
    }

    # ---------- Aggregate similarity scores ----------
    similarity_scores = np.zeros(len(movies_df))

    for movie_id in user_movies:
        idx = movie_id_to_index[movie_id]
        similarity_scores += cosine_sim[idx]

    # Average similarity
    similarity_scores /= len(user_movies)

    # ---------- Build recommendation DataFrame ----------
    recs_df = movies_df.copy()
    recs_df["similarity_score"] = similarity_scores

    # Exclude already watched movies
    recs_df = recs_df[~recs_df["movie_id"].isin(user_movies)]

    # Sort by similarity score (descending)
    recs_df = recs_df.sort_values(
        by="similarity_score",
        ascending=False
    )

    # Select top-N
    recs_df = recs_df.head(top_n)

    # Keep EXACT required columns
    recs_df = recs_df[["movie_id", "title", "similarity_score"]]

    # ---------- Save output ----------
    recs_df.to_csv(CONTENT_OUT, index=False)

    print(f"Content-based recommendations saved to: {CONTENT_OUT}")


if __name__ == "__main__":
    content_based_recommendations()
