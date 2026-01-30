import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from src.config import (
    PROCESSED_DATA,
    OUTPUT_DIR,
    MODEL_DIR,
    SVD_OUT
)


def train_svd_model(user_id: int = 1, top_n: int = 10):
    """
    Train an SVD model using Surprise, save the trained model,
    save movie metadata, and generate top-N recommendations
    for a given user.
    """

    # ---------- Ensure output directories exist ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load processed dataset ----------
    df = pd.read_csv(PROCESSED_DATA)

    # ---------- Prepare Surprise dataset ----------
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    # ---------- Train SVD model ----------
    svd_model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    svd_model.fit(trainset)

    # ---------- Save trained SVD model ----------
    svd_model_path = MODEL_DIR / "svd.pkl"
    with open(svd_model_path, "wb") as f:
        pickle.dump(svd_model, f)

    # ---------- Save movie metadata ----------
    movie_info = (
        df[["movie_id", "title"]]
        .drop_duplicates()
        .set_index("movie_id")
    )

    movie_info_path = MODEL_DIR / "movie_info.pkl"
    with open(movie_info_path, "wb") as f:
        pickle.dump(movie_info, f)

    # ---------- Generate recommendations ----------
    rated_movies = set(
        df[df["user_id"] == user_id]["movie_id"].unique()
    )

    all_movie_ids = df["movie_id"].unique()

    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movies:
            pred = svd_model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    # ---------- Build output DataFrame ----------
    recs_df = pd.DataFrame(
        top_predictions,
        columns=["movie_id", "estimated_rating"]
    )

    recs_df["title"] = recs_df["movie_id"].map(movie_info["title"])

    # Reorder columns EXACTLY as required
    recs_df = recs_df[["movie_id", "title", "estimated_rating"]]

    # ---------- Save recommendations ----------
    recs_df.to_csv(SVD_OUT, index=False)

    print("SVD model trained successfully")
    print(f"Model saved to: {svd_model_path}")
    print(f"Movie info saved to: {movie_info_path}")
    print(f"Recommendations saved to: {SVD_OUT}")


if __name__ == "__main__":
    train_svd_model()
