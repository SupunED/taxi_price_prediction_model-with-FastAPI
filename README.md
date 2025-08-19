# Taxi Price Prediction API

## Overview
This repository contains a machine learning project that predicts taxi trip prices.  
The project consists of two main parts:
- A **Jupyter Notebook** for model training.
- A **FastAPI application** that exposes the trained model as a web API.

---

## Project Structure
- **`model_traning.ipynb`**  
  Jupyter Notebook detailing the training process, including:
  - Data loading
  - Exploratory Data Analysis (EDA)
  - Handling missing values
  - Training the final model

- **`main.py`**  
  Python script that sets up a RESTful API using FastAPI.  
  - Loads the trained machine learning model  
  - Provides an endpoint to receive trip features and return a predicted price  

- **`Model/`**  
  Directory to hold the trained machine learning model file:  
  `Taxi_Price_Prediction_Model.pkl`

---

## API Endpoints

### **`/predict`** (POST)
Accepts a JSON object with trip details and returns the predicted price.  

#### Input Schema (`PredictionInput`)
| Feature               | Data Type | Description |
|------------------------|-----------|-------------|
| `Trip_Distance_km`     | float     | The distance of the trip in kilometers. |
| `Time_of_Day`          | int       | Categorical: 0=Morning, 1=Afternoon, 2=Evening, 3=Night. |
| `Day_of_Week`          | int       | Categorical: 0=Weekday, 1=Weekend. |
| `Passenger_Count`      | int       | The number of passengers. |
| `Traffic_Conditions`   | int       | Categorical: 0=Low, 1=Medium, 2=High. |
| `Weather`              | int       | Categorical: 0=Clear, 1=Rain, 2=Snow. |
| `Base_Fare`            | float     | The initial base fare of the trip. |
| `Per_Km_Rate`          | float     | The price per kilometer. |
| `Per_Minute_Rate`      | float     | The price per minute. |
| `Trip_Duration_Minutes`| float     | The duration of the trip in minutes. |

#### Output Schema (`PredictionOutput`)
| Feature           | Data Type | Description |
|--------------------|-----------|-------------|
| `predicted_price` | float     | The predicted taxi trip price. |

---

### **`/model-info`** (GET)
Returns information about the loaded model.

---

## Model Details
- Model: **Random Forest Regressor**
- Dataset: Historical taxi trip data
- Reason: Chosen for its high accuracy in predicting trip prices  
- Training details available in: `model_traning.ipynb`
- Model file: `Model/Taxi_Price_Prediction_Model.pkl`

---

## How to Run the API

1. **Clone the repository**
   ```bash
   git clone https://github.com/SupunED/taxi_price_prediction_model-with-FastAPI.git
   ```
2. **Install dependencies**
   ```bash
   pip install requirements.txt
   ```
3. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```
4. The API will be accessible at:
   ```
   http://127.0.0.1:8000
   ```
