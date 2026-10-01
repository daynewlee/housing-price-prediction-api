import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
TARGET_COLUMN = "price"

def train():
    print("[1/3] Loading training dataset...")
    df = pd.read_csv("data/House-Price-Dataset.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[2/3] Fitting Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Validation R2 Score: {r2_score(y_test, y_pred):.4f}")

    print("[3/3] Exporting model to artifacts/housing_model.joblib...")
    joblib.dump(model, "artifacts/housing_model.joblib")
    print("Training pipeline finished successfully.")

if __name__ == "__main__":
    train()
