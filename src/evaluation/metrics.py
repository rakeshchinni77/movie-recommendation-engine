import json
import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
from surprise.model_selection import train_test_split
from src.config import PROCESSED_DATA, EVAL_OUT, OUTPUT_DIR


# ---------- Ranking Metrics ----------

def precision_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        relevant = sum((true_r >= threshold) for (_, true_r) in top_k)
        precisions.append(relevant / k)

    return np.mean(precisions)


def ndcg_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    ndcgs = []

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        dcg = sum(
            (1 / np.log2(i + 2))
            for i, (_, true_r) in enumerate(top_k)
            if true_r >= threshold
        )

        ideal = sorted(
            [true_r for (_, true_r) in user_ratings if true_r >= threshold],
            reverse=True
        )[:k]

        idcg = sum(1 / np.log2(i + 2) for i in range(len(ideal)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcgs)


# ---------- Main Evaluation ----------

def evaluate_models():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["user_id", "movie_id", "rating"]],
        reader
    )

    # CORRECT Surprise API
    trainset, testset = train_test_split(
        data, test_size=0.2, random_state=42
    )

    # ---------- User-Based CF ----------
    sim_options = {"name": "cosine", "user_based": True}
    user_cf = KNNBasic(sim_options=sim_options)
    user_cf.fit(trainset)
    user_preds = user_cf.test(testset)

    user_rmse = accuracy.rmse(user_preds, verbose=False)
    user_prec = precision_at_k(user_preds)
    user_ndcg = ndcg_at_k(user_preds)

    # ---------- SVD ----------
    svd = SVD(random_state=42)
    svd.fit(trainset)
    svd_preds = svd.test(testset)

    svd_rmse = accuracy.rmse(svd_preds, verbose=False)
    svd_prec = precision_at_k(svd_preds)
    svd_ndcg = ndcg_at_k(svd_preds)

    # ---------- Save JSON ----------
    metrics = {
        "user_based_cf": {
            "rmse": round(float(user_rmse), 4),
            "precision_at_10": round(float(user_prec), 4),
            "ndcg_at_10": round(float(user_ndcg), 4)
        },
        "svd": {
            "rmse": round(float(svd_rmse), 4),
            "precision_at_10": round(float(svd_prec), 4),
            "ndcg_at_10": round(float(svd_ndcg), 4)
        }
    }

    with open(EVAL_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics saved to: {EVAL_OUT}")


if __name__ == "__main__":
    evaluate_models()
