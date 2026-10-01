from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db, PropertyEstimateRecord
from app.schemas import (
    PropertyEstimateCreate,
    PropertyEstimateItem,
    EstimateHistoryResponse,
    ComparisonRequest,
    ComparisonResponse
)

router = APIRouter(prefix="/api/estimates", tags=["Property Value Estimator"])

@router.post("/", response_model=PropertyEstimateItem, status_code=status.HTTP_201_CREATED)
def submit_estimate(
    payload: PropertyEstimateCreate,
    db: Session = Depends(get_db)
):
    """
    Submits property details from frontend form, performs regression inference,
    and persists the calculated estimate into SQLite.
    """
    from app.main import ml_models, FEATURE_ORDER  # Import loaded ML model
    
    model = ml_models.get("housing_model")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is currently not loaded."
        )

    # 1. Feature extraction in strict order expected by scikit-learn
    feature_vector = [[getattr(payload, col) for col in FEATURE_ORDER]]

    # 2. Model inference
    raw_prediction = float(model.predict(feature_vector)[0])
    calculated_price = round(max(0.0, raw_prediction), 2)

    # 3. Persist to SQLite
    record = PropertyEstimateRecord(
        property_name=payload.property_name or "Untitled Property",
        square_footage=payload.square_footage,
        bedrooms=payload.bedrooms,
        bathrooms=payload.bathrooms,
        year_built=payload.year_built,
        lot_size=payload.lot_size,
        distance_to_city_center=payload.distance_to_city_center,
        school_rating=payload.school_rating,
        predicted_price=calculated_price,
        currency="USD"
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return record

@router.get("/history", response_model=EstimateHistoryResponse)
def get_estimate_history(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Retrieves previous estimates sorted by creation date (newest first).
    """
    records = (
        db.query(PropertyEstimateRecord)
        .order_by(PropertyEstimateRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "total_records": len(records),
        "records": records
    }

@router.post("/compare", response_model=ComparisonResponse)
def compare_properties(
    payload: ComparisonRequest,
    db: Session = Depends(get_db)
):
    """
    Fetches multiple properties by IDs for side-by-side frontend comparison.
    """
    records = (
        db.query(PropertyEstimateRecord)
        .filter(PropertyEstimateRecord.id.in_(payload.estimate_ids))
        .all()
    )
    if len(records) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 valid property records must be found to compare."
        )
    return {"properties": records}
