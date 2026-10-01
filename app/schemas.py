from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

# ============================================================================
# 1. Base Machine Learning Inference Schemas (Task 1 / Core Model)
# ============================================================================

class HousingFeatures(BaseModel):
    square_footage: float = Field(..., gt=0, description="Total interior living space in sq ft")
    bedrooms: int = Field(..., ge=0, description="Total number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    year_built: int = Field(..., gt=1800, description="Construction year")
    lot_size: float = Field(..., gt=0, description="Total lot area in sq ft")
    distance_to_city_center: float = Field(..., ge=0, description="Distance to city center in miles/km")
    school_rating: float = Field(..., ge=0, le=10, description="Local school rating (0.0 to 10.0)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "square_footage": 1550.0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "year_built": 1997,
                "lot_size": 6800.0,
                "distance_to_city_center": 4.1,
                "school_rating": 7.6
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated residential market value in USD")
    currency: str = Field(default="USD")
    status: str = Field(default="success")

class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[float]
    status: str = Field(default="success")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================================
# 2. Property Value Estimator & Database Schemas (Task 2 / Next.js Service)
# ============================================================================

class PropertyEstimateCreate(BaseModel):
    property_name: Optional[str] = Field(default="My Property", description="User-assigned label for comparison")
    square_footage: float = Field(..., gt=0, description="Gross living area in sq ft")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    year_built: int = Field(..., gt=1800, description="Year of construction")
    lot_size: float = Field(..., gt=0, description="Lot size in sq ft")
    distance_to_city_center: float = Field(..., ge=0, description="Distance in miles/km")
    school_rating: float = Field(..., ge=0, le=10, description="Local school rating (0.0 to 10.0)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "property_name": "Sample Suburban House",
                "square_footage": 1850.0,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "year_built": 2005,
                "lot_size": 7500.0,
                "distance_to_city_center": 5.2,
                "school_rating": 8.0
            }
        }
    }

class PropertyEstimateItem(PropertyEstimateCreate):
    id: int
    predicted_price: float
    currency: str
    created_at: datetime

    model_config = {"from_attributes": True}

class EstimateHistoryResponse(BaseModel):
    total_records: int
    records: List[PropertyEstimateItem]

class ComparisonRequest(BaseModel):
    estimate_ids: List[int] = Field(..., min_length=2, max_length=5, description="IDs of 2 to 5 estimates to compare")

class ComparisonResponse(BaseModel):
    properties: List[PropertyEstimateItem]
