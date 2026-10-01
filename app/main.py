from contextlib import asynccontextmanager
import os
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import estimator
from app.schemas import (
    HousingFeatures,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)

# 1. Global constants and state storage
MODEL_PATH = "artifacts/housing_model.joblib"
FEATURE_ORDER = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating"
]
ml_models = {}

# 2. Application lifespan (Database initialization & Model pre-warming)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database schema
    init_db()
    
    # Load ML Model weights into memory once
    if os.path.exists(MODEL_PATH):
        ml_models["housing_model"] = joblib.load(MODEL_PATH)
    else:
        ml_models["housing_model"] = None
        
    yield
    ml_models.clear()

# 3. Create FastAPI instance
app = FastAPI(
    title="Housing Price Prediction Model API",
    description="Machine Learning REST API for residential valuation, real-time inference, and property estimate tracking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 4. Configure CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Register modular routers
app.include_router(estimator.router)

# 6. Core diagnostic and legacy inference routes
@app.get("/", tags=["General"])
def root():
    return {
        "service": "Housing Price Prediction Model API",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    loaded = ml_models.get("housing_model") is not None
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded}

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_single(features: HousingFeatures):
    model = ml_models.get("housing_model")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train and export artifacts first."
        )

    vector = [[getattr(features, col) for col in FEATURE_ORDER]]
    pred = float(model.predict(vector)[0])
    return {
        "predicted_price": round(pred, 2),
        "currency": "USD",
        "status": "success"
    }

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_batch(items: list[HousingFeatures]):
    model = ml_models.get("housing_model")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    vectors = [[getattr(item, col) for col in FEATURE_ORDER] for item in items]
    raw_preds = model.predict(vectors)
    return {
        "total_records": len(items),
        "predictions": [round(float(p), 2) for p in raw_preds],
        "status": "success"
    }
