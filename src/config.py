from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"

# Processed data
PROCESSED_DATA = DATA_DIR / "processed_movies.csv"

# Recommendation outputs
USER_CF_OUT = OUTPUT_DIR / "user_based_recommendations.csv"
SVD_OUT = OUTPUT_DIR / "svd_recommendations.csv"
CONTENT_OUT = OUTPUT_DIR / "content_based_recommendations.csv"
COLD_START_OUT = OUTPUT_DIR / "cold_start_recommendations.csv"

# Evaluation output
EVAL_OUT = OUTPUT_DIR / "evaluation_metrics.json"
