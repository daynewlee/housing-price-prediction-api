import pandas as pd
import joblib

FEATURE_COLUMNS = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating"
]

def run_test_prediction():
    model = joblib.load("artifacts/housing_model.joblib")
    df = pd.read_csv("data/Test-Data-Prediction.csv")
    
    predictions = model.predict(df[FEATURE_COLUMNS])
    df["predicted_price"] = [round(p, 2) for p in predictions]
    
    output_path = "artifacts/Test-Data-Predicted-Results.csv"
    df.to_csv(output_path, index=False)
    print(f"Batch prediction done. Results saved to {output_path}")

if __name__ == "__main__":
    run_test_prediction()
