import pickle
import pandas as pd
from fastapi import FastAPI
from src.config import (
    MODEL_DIR,
    PROCESSED_DATA,
    COLD_START_OUT
)

app = FastAPI(title="Movie Recommendation Engine")

# ---------- Load resources at startup ----------
with open(MODEL_DIR / "svd.pkl", "rb") as f:
    svd_model = pickle.load(f)

with open(MODEL_DIR / "movie_info.pkl", "rb") as f:
    movie_info = pickle.load(f)

df = pd.read_csv(PROCESSED_DATA)

# Cache cold-start recommendations
cold_start_df = pd.read_csv(COLD_START_OUT)


# ---------- Health Check ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------- Recommendations Endpoint ----------
@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):
    """
    Return top-10 movie recommendations for a given user.
    Uses SVD if user exists, otherwise falls back to cold-start.
    """

    # ---------- Check if user exists ----------
    user_exists = user_id in df["user_id"].unique()

    recommendations = []

    if user_exists:
        # Movies already rated by user
        rated_movies = set(
            df[df["user_id"] == user_id]["movie_id"].unique()
        )

        all_movie_ids = df["movie_id"].unique()
        predictions = []

        for movie_id in all_movie_ids:
            if movie_id not in rated_movies:
                pred = svd_model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))

        # Sort and take top 10
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:10]

        for movie_id, est_rating in top_predictions:
            recommendations.append({
                "movie_id": int(movie_id),
                "title": movie_info.loc[movie_id, "title"],
                "estimated_rating": round(float(est_rating), 3)
            })

    else:
        # ---------- Cold-start fallback ----------
        for _, row in cold_start_df.iterrows():
            recommendations.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "estimated_rating": round(float(row["average_rating"]), 3)
            })

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }
