import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA, DATA_DIR


def preprocess_movielens():
    """
    Preprocess MovieLens 100k dataset and create a cleaned CSV
    with the following schema:

    user_id | movie_id | rating | title | genres
    """

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    #Load ratings data (u.data)
    ratings_path = RAW_DATA_DIR / "u.data"
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    # Drop timestamp (not needed)
    ratings = ratings.drop(columns=["timestamp"])

    #Load movie metadata (u.item)
    movies_path = RAW_DATA_DIR / "u.item"

    # MovieLens genre columns (fixed order)
    genre_columns = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    movie_columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url"
    ] + genre_columns

    movies = pd.read_csv(
        movies_path,
        sep="|",
        names=movie_columns,
        encoding="latin-1"
    )

    # Convert one-hot genre columns into pipe-separated string
    def extract_genres(row):
        return "|".join([genre for genre in genre_columns if row[genre] == 1])

    movies["genres"] = movies.apply(extract_genres, axis=1)

    # Keep only required columns
    movies = movies[["movie_id", "title", "genres"]]

    # Merge ratings with movie metadata
    merged_df = pd.merge(
        ratings,
        movies,
        on="movie_id",
        how="inner"
    )

    #Save processed CSV
    merged_df.to_csv(PROCESSED_DATA, index=False)

    print(f"Processed data saved to: {PROCESSED_DATA}")


if __name__ == "__main__":
    preprocess_movielens()
