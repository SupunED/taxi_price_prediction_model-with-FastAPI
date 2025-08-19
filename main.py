from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained RandomForest model
try:
    model = joblib.load("Model/Taxi_Price_Prediction_Model.pkl")  # put file in same folder as main.py
except Exception as e:
    raise RuntimeError(f"Model could not be loaded: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Taxi Price Prediction API",
    description="Predict taxi trip prices based on trip and contextual features",
    version="1.0.0"
)

# ==== Input Schema ====
class PredictionInput(BaseModel):
    Trip_Distance_km: float
    Time_of_Day: int        # 0=Morning, 1=Afternoon, 2=Evening, 3=Night
    Day_of_Week: int        # 0=Weekday, 1=Weekend
    Passenger_Count: int
    Traffic_Conditions: int # 0=Low, 1=Medium, 2=High
    Weather: int            # 0=Clear, 1=Rain, 2=Snow
    Base_Fare: float
    Per_Km_Rate: float
    Per_Minute_Rate: float
    Trip_Duration_Minutes: float

# ==== Output Schema ====
class PredictionOutput(BaseModel):
    predicted_price: float


# ==== Endpoints ====
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Taxi Price Prediction API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert request data into numpy array
        features = np.array([[
            input_data.Trip_Distance_km,
            input_data.Time_of_Day,
            input_data.Day_of_Week,
            input_data.Passenger_Count,
            input_data.Traffic_Conditions,
            input_data.Weather,
            input_data.Base_Fare,
            input_data.Per_Km_Rate,
            input_data.Per_Minute_Rate,
            input_data.Trip_Duration_Minutes
        ]])

        # Run prediction
        prediction = model.predict(features)[0]

        return PredictionOutput(predicted_price=float(prediction))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestRegressor",
        "problem_type": "regression",
        "features": [
            "Trip_Distance_km", "Time_of_Day", "Day_of_Week", "Passenger_Count",
            "Traffic_Conditions", "Weather", "Base_Fare", "Per_Km_Rate",
            "Per_Minute_Rate", "Trip_Duration_Minutes"
        ]
    }
