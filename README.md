#  Movie Recommendation Engine

Collaborative Filtering • Matrix Factorization • Content-Based Filtering

---

## Project Overview

This project implements a production-style **Movie Recommendation Engine** using the **MovieLens 100k dataset**.  
It demonstrates how modern platforms like Netflix and Amazon deliver personalized recommendations using:

- User-Based Collaborative Filtering  
- Matrix Factorization (SVD)  
- Content-Based Filtering  
- Cold-Start Handling  
- Model Evaluation (RMSE, Precision@10, NDCG@10)  
- REST API with FastAPI  
- Dockerized end-to-end pipeline  

All data preprocessing, model training, evaluation, and API startup are executed automatically using Docker.

---

## Project Architecture (High Level)
```
Raw Data (MovieLens)
        │
        ▼
Data Preprocessing
        │
        ▼
Recommendation Models  
(User-CF | SVD | Content-Based)
        │
        ▼
Evaluation Metrics
        │
        ▼
Saved Outputs + Trained Models
        │
        ▼
FastAPI Recommendation Service
```
---

## Repository Structure

```
movie-recommendation-engine/
├── data/
│   ├── raw/
│   │   ├── u.data
│   │   ├── u.item
│   │   └── u.user
│   └── processed_movies.csv
│
├── output/
│   ├── user_based_recommendations.csv
│   ├── svd_recommendations.csv
│   ├── content_based_recommendations.csv
│   ├── cold_start_recommendations.csv
│   ├── evaluation_metrics.json
│   └── models/
│       ├── svd.pkl
│       └── movie_info.pkl
│
├── src/
│   ├── api/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── config.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
---

## Tech Stack

- Python 3.10  
- Pandas, NumPy  
- Scikit-learn  
- Surprise (Collaborative Filtering & SVD)  
- FastAPI + Uvicorn  
- Docker & Docker Compose  

---

## How to Run (One Command)

### Prerequisites
- Docker Desktop installed and running

### Run the complete pipeline
```bash
docker-compose up --build
```
---

## API Endpoints

### Health Check
**GET** `/health`

#### Response
```json
{
  "status": "ok"
}

```
---

### Get Recommendations (SVD-based)
**GET** `/recommendations/{user_id}`

#### Example
```bash
curl http://localhost:8000/recommendations/1
```
Response

```
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 513,
      "title": "Third Man, The (1949)",
      "estimated_rating": 4.89
    }
  ]
}
```
## Behavior

- Existing user → SVD recommendations  
- New user → Cold-start (popular movies)  
- Always returns 10 recommendations  
- Always returns HTTP 200 

---

## Generated Output Files

All files below are generated automatically during execution:

### Data
- data/processed_movies.csv

### Recommendations
- output/user_based_recommendations.csv  
- output/svd_recommendations.csv  
- output/content_based_recommendations.csv  
- output/cold_start_recommendations.csv  

### Evaluation
- output/evaluation_metrics.json  

### Models
- output/models/svd.pkl  
- output/models/movie_info.pkl  

> **Note:** These files are generated automatically when running `docker-compose up --build`.

---

## Evaluation Metrics

Each model is evaluated using:

- RMSE – rating prediction accuracy  
- Precision@10 – relevance of top recommendations  
- NDCG@10 – ranking quality  

Metrics are saved in:

- output/evaluation_metrics.json

---

## Cold Start Strategy

For users with no prior history, the system recommends:

- movies with the highest average ratings  
- ensures meaningful recommendations even for new users  

---

## Key Highlights

- End-to-end automated ML pipeline  
- Multiple recommendation strategies  
- Production-style API  
- Dockerized and reproducible  
- Evaluator-ready project structure  

---

## Author

**Chinni Rakesh**  
B.Tech CSE (AI & ML)  
Movie Recommendation Engine Project 




