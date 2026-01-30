import pandas as pd
from src.config import (
    PROCESSED_DATA,
    COLD_START_OUT,
    OUTPUT_DIR
)


def generate_cold_start_recommendations(top_n: int = 10):
    """
    Generate cold-start recommendations based on
    highest average movie ratings.
    """

    # ---------- Ensure output directory exists ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load processed data ----------
    df = pd.read_csv(PROCESSED_DATA)

    # ---------- Compute average rating per movie ----------
    avg_ratings = (
        df.groupby(["movie_id", "title"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "average_rating"})
    )

    # ---------- Sort by average rating (descending) ----------
    avg_ratings = avg_ratings.sort_values(
        by="average_rating",
        ascending=False
    )

    # ---------- Select Top-N ----------
    top_movies = avg_ratings.head(top_n)

    # ---------- Keep EXACT required columns ----------
    top_movies = top_movies[["movie_id", "title", "average_rating"]]

    # ---------- Save output ----------
    top_movies.to_csv(COLD_START_OUT, index=False)

    print(f"Cold-start recommendations saved to: {COLD_START_OUT}")


if __name__ == "__main__":
    generate_cold_start_recommendations()
