from src.data.preprocessing import preprocess_movielens
from src.models.user_cf import train_user_based_cf
from src.models.svd import train_svd_model
from src.models.content_based import content_based_recommendations
from src.models.cold_start import generate_cold_start_recommendations
from src.evaluation.metrics import evaluate_models


def run_pipeline():
    """
    Orchestrates the full Movie Recommendation Engine pipeline.

    This function is intentionally sequential and explicit so that:
    docker-compose up -> everything runs automatically.
    """

    print("\n Starting Movie Recommendation Engine pipeline...\n")

    # ---------- Step 1: Data Preprocessing ----------
    print("Step 1: Preprocessing MovieLens data")
    preprocess_movielens()

    # ---------- Step 2: User-Based Collaborative Filtering ----------
    print("Step 2: Training User-Based Collaborative Filtering model")
    train_user_based_cf(user_id=1, top_n=10)

    # ---------- Step 3: Matrix Factorization (SVD) ----------
    print("Step 3: Training SVD Matrix Factorization model")
    train_svd_model(user_id=1, top_n=10)

    # ---------- Step 4: Content-Based Filtering ----------
    print("Step 4: Generating Content-Based recommendations")
    content_based_recommendations(user_id=1, top_n=10)

    # ---------- Step 5: Cold-Start Recommendations ----------
    print("Step 5: Generating Cold-Start recommendations")
    generate_cold_start_recommendations(top_n=10)

    # ---------- Step 6: Model Evaluation ----------
    print("Step 6: Evaluating models (RMSE, Precision@10, NDCG@10)")
    evaluate_models()

    print("\nPipeline execution completed successfully.")
    print("All outputs saved to the output/ directory\n")


if __name__ == "__main__":
    run_pipeline()
