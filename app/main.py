from contextlib import asynccontextmanager
import os
import joblib
from fastapi import FastAPI, HTTPException, status
from app.schemas import HousingFeatures, PredictionResponse, BatchPredictionResponse, HealthResponse

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        ml_models["housing_model"] = joblib.load(MODEL_PATH)
    else:
        ml_models["housing_model"] = None
    yield
    ml_models.clear()

app = FastAPI(
    title="Housing Price Prediction Model API",
    description="Machine Learning REST API for real-time and batch residential valuation. Built with FastAPI and Scikit-Learn.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

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
