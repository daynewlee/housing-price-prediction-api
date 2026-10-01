from pydantic import BaseModel, Field

class HousingFeatures(BaseModel):
    square_footage: float = Field(..., gt=0, description="Total interior living space in sq ft", example=1550.0)
    bedrooms: int = Field(..., ge=0, description="Total number of bedrooms", example=3)
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms", example=2.0)
    year_built: int = Field(..., gt=1800, description="Construction year", example=1997)
    lot_size: float = Field(..., gt=0, description="Total lot area in sq ft", example=6800.0)
    distance_to_city_center: float = Field(..., ge=0, description="Distance to city center in miles/km", example=4.1)
    school_rating: float = Field(..., ge=0, le=10, description="Local school rating (0.0 to 10.0)", example=7.6)

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
    predicted_price: float = Field(..., description="Estimated residential market value in USD", example=248500.50)
    currency: str = Field(default="USD", example="USD")
    status: str = Field(default="success", example="success")

class BatchPredictionResponse(BaseModel):
    total_records: int = Field(..., example=4)
    predictions: list[float]
    status: str = Field(default="success", example="success")

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
