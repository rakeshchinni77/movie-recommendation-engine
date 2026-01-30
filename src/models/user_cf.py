import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from src.config import (
    PROCESSED_DATA,
    USER_CF_OUT,
    OUTPUT_DIR
)


def train_user_based_cf(user_id: int = 1, top_n: int = 10):
    """
    Train a user-based collaborative filtering model using Surprise (KNN)
    and generate top-N recommendations for a given user.
    """

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load processed data ----------
    df = pd.read_csv(PROCESSED_DATA)

    # Surprise expects: user, item, rating
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    # ---------- Configure User-Based KNN ----------
    sim_options = {
        "name": "cosine",
        "user_based": True
    }

    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)

    # ---------- Get movies already rated by the user ----------
    rated_movies = set(
        df[df["user_id"] == user_id]["movie_id"].unique()
    )

    # ---------- Predict ratings for unseen movies ----------
    all_movie_ids = df["movie_id"].unique()

    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movies:
            pred = model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

    # ---------- Select Top-N recommendations ----------
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    # ---------- Build output DataFrame ----------
    recs_df = pd.DataFrame(
        top_predictions,
        columns=["movie_id", "estimated_rating"]
    )

    # Add movie titles
    movie_titles = (
        df[["movie_id", "title"]]
        .drop_duplicates()
        .set_index("movie_id")
    )

    recs_df["title"] = recs_df["movie_id"].map(
        movie_titles["title"]
    )

    # Reorder columns EXACTLY as required
    recs_df = recs_df[["movie_id", "title", "estimated_rating"]]

    # ---------- Save output ----------
    recs_df.to_csv(USER_CF_OUT, index=False)

    print(f"User-based CF recommendations saved to: {USER_CF_OUT}")


if __name__ == "__main__":
    train_user_based_cf()
