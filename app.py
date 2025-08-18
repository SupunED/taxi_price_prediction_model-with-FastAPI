import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# The `try-except` block attempts to load the real model first.
try:
    model = joblib.load("Model/Taxi_Price_Prediction_Model.pkl")
    print("Successfully loaded the pre-trained model.")
except FileNotFoundError:
    print("ML Model not found")

# Define the input data model using Pydantic for data validation.
class TaxiTrip(BaseModel):
    Trip_Distance_km: float
    Time_of_Day: str
    Day_of_Week: str
    Passenger_Count: float
    Traffic_Conditions: str
    Weather: str
    Base_Fare: float
    Per_Km_Rate: float
    Per_Minute_Rate: float
    Trip_Duration_Minutes: float

# Initialize the FastAPI application
app = FastAPI(
    title="Taxi Price Prediction API",
    description="An API to predict taxi trip prices using a pre-trained model."
)

# Manually defined mappings for categorical features
# These mappings are based on the unique values found in the exploratory data analysis.
time_of_day_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
day_of_week_mapping = {'Weekday': 0, 'Weekend': 1}
traffic_conditions_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
weather_mapping = {'Clear': 0, 'Rain': 1, 'Snow': 2, 'Stormy': 3}

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("client.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict_price(trip: TaxiTrip) -> Dict[str, Any]:
    """
    Predicts the taxi trip price based on the input features.

    Args:
        trip (TaxiTrip): A Pydantic model containing all the trip details.

    Returns:
        Dict[str, Any]: A dictionary containing the predicted price.
    """
    try:
        # Preprocess the categorical features based on the mappings.
        # This step is crucial for the model to correctly interpret the data.
        processed_data = {
            'Trip_Distance_km': trip.Trip_Distance_km,
            'Time_of_Day': time_of_day_mapping.get(trip.Time_of_Day.title()),
            'Day_of_Week': day_of_week_mapping.get(trip.Day_of_Week.title()),
            'Passenger_Count': trip.Passenger_Count,
            'Traffic_Conditions': traffic_conditions_mapping.get(trip.Traffic_Conditions.title()),
            'Weather': weather_mapping.get(trip.Weather.title()),
            'Base_Fare': trip.Base_Fare,
            'Per_Km_Rate': trip.Per_Km_Rate,
            'Per_Minute_Rate': trip.Per_Minute_Rate,
            'Trip_Duration_Minutes': trip.Trip_Duration_Minutes
        }

        # Create a pandas DataFrame from the processed data.
        # It's important to match the column order from the training phase.
        input_df = pd.DataFrame([processed_data])

        # Make the prediction using the loaded model.
        prediction = model.predict(input_df)

        # Return the prediction in a JSON response.
        return {"predicted_price": prediction[0]}

    except KeyError as e:
        # Handle cases where a categorical value is not in the defined mappings.
        return {"error": f"Invalid value for a categorical feature: {e}. Please check your input."}
    except Exception as e:
        # General error handling for any other issues during prediction.
        return {"error": f"An error occurred during prediction: {e}"}

