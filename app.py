from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the serialized model
model = joblib.load("../model-22-09-2024-13-35-20-700061.pkl")

app = FastAPI()

# Define the input data structure based on the model's expected input
class SalesInput(BaseModel):
    customers: int
    open: int
    promo: int
    day_of_week: int
    state_holiday: int
    school_holiday: int
    weekday: int
    weekend: int
    days_to_holiday: int
    days_after_holiday: int
    beginning_of_month: int
    mid_month: int
    end_of_month: int
    month: int
    is_holiday: int
    store: int

@app.post("/predict/")
def predict_sales(data: SalesInput):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Return the prediction
        return {"predicted_sales": prediction[0]}
    except Exception as e:
        return {"error": str(e)}


import requests

url = "http://127.0.0.1:8000/predict/"

data = {
    "customers": 100,
    "open": 1,
    "promo": 1,
    "day_of_week": 5,
    "state_holiday": 0,
    "school_holiday": 1,
    "weekday": 1,
    "weekend": 0,
    "days_to_holiday": 10,
    "days_after_holiday": 5,
    "beginning_of_month": 1,
    "mid_month": 0,
    "end_of_month": 0,
    "month": 9,
    "is_holiday": 0,
    "store": 1
}

response = requests.post(url, json=data)

print(response.json())



























# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# # Load the serialized pipeline
# pipeline = joblib.load("../model-22-09-2024-13-35-20-700061.pkl")  # Replace with your actual pipeline filename

# app = FastAPI()

# # Define input data structure
# class SalesInput(BaseModel):
#     previous_sales: list
#     customers: int = 0
#     open: int = 1
#     weekday: int = 1
#     weekend: int = 0
#     days_to_holiday: int = 5
#     days_after_holiday: int = 0
#     beginning_of_month: int = 0
#     mid_month: int = 0
#     end_of_month: int = 0
#     promo: int = 0
#     state_holiday: int = 0
#     school_holiday: int = 0
#     month: int = 1
#     store: int = 1
#     is_holiday: int = 0
#     day_of_week: int = 1

# @app.post("/predict/")
# def predict_sales(data: SalesInput):
#     try:
#         # Create a DataFrame with the input data
#         input_data = pd.DataFrame(columns=['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo',
#                                            'StateHoliday', 'SchoolHoliday', 'Weekday', 'Weekend',
#                                            'DaysToHoliday', 'DaysAfterHoliday', 'BeginningOfMonth',
#                                            'MidMonth', 'EndOfMonth', 'Month', 'IsHoliday'])

#         # Populate the DataFrame with the provided data
#         input_data.loc[0] = [
#             data.store,
#             data.day_of_week,
#             data.customers,
#             data.open,
#             data.promo,
#             data.state_holiday,
#             data.school_holiday,
#             data.weekday,
#             data.weekend,
#             data.days_to_holiday,
#             data.days_after_holiday,
#             data.beginning_of_month,
#             data.mid_month,
#             data.end_of_month,
#             data.month,
#             data.is_holiday,
#         ]

#         # Make prediction using the pipeline
#         prediction = pipeline.predict(input_data)

#         # Return the prediction
#         return {"predicted_sales": prediction[0]}  # Adjust based on your model's output shape
#     except Exception as e:
#         return {"error": str(e)}

# # model_serving_api.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware

# # Initialize the FastAPI app
# app = FastAPI()

# # Allow CORS for external access if needed
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to specific domains in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"-],
# )

# # Load the serialized model using pickle
# with open('../model-22-09-2024-13-35-20-700061.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Pydantic model for input validation
# class Features(BaseModel):
#     features: list

# @app.get("/")
# def root():
#     return {"message": "Welcome to the Model Serving API"}

# # Define an API endpoint for predictions
# @app.post("/predict")
# def predict(input_data: Features):
#     # Convert the input into a NumPy array or as needed by your model
#     features_array = np.array([input_data.features])
    
#     # Make the prediction using the loaded model
#     prediction = model.predict(features_array)
    
#     # Return the prediction as a response
#     return {"prediction": prediction.tolist()}



# # model_serving_api.py
# from fastapi import FastAPI
# import pickle
# import numpy as np

# # Initialize the FastAPI app
# app = FastAPI()

# # Load the serialized model using pickle
# with open('../model-22-09-2024-13-35-20-700061.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# @app.get("/")
# def root():
#     return {"message": "Welcome to the Model Serving API"}

# # Define an API endpoint for predictions
# @app.post("/predict")
# def predict(features: list):
#     # Convert the input into a NumPy array or as needed by your model
#     features_array = np.array([features])
    
#     # Make the prediction using the loaded model
#     prediction = model.predict(features_array)
    
#     # Return the prediction as a response
#     return {"prediction": prediction.tolist()}
