import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Assuming process_data and inference are in ml.data and ml.model respectively
from ml.data import process_data
from ml.model import inference, load_model

#  Data Model Definition 
# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

    # This allows the model to be created from a dictionary with hyphenated keys
    class Config:
        allow_population_by_field_name = True


#  Load Model and Encoder 
# This part of the code runs once when the script is first loaded.
try:
    # Path for the saved encoder
    encoder_path = "model/encoder.pkl"
    encoder = load_model(encoder_path)

    # Path for the saved label binarizer
    lb_path = "model/lb.pkl"
    lb = load_model(lb_path)

    # Path for the saved model
    model_path = "model/model.pkl"
    model = load_model(model_path)
    print("Model, encoder, and label binarizer loaded successfully.")

except FileNotFoundError:
    print("Error: Model or processor file not found. Please run train_model.py first.")
    # Exit if files are not found, as the API cannot function.
    exit()


#  API Creation 
# Create a RESTful API using FastAPI
app = FastAPI(
    title="Census Income Prediction API",
    description="An API to predict whether an individual's income exceeds $50K.",
    version="1.0.0",
)


#  API Endpoints 
# Create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello and provide basic API information. """
    return {"message": "Welcome to the Census Income Prediction API!"}


# Create a POST on a different path that does model inference
# The path is changed to /predict for clarity, as is common practice.
@app.post("/predict")
async def post_inference(data: Data):
    """
    Receives census data for an individual and returns a prediction for their income level.
    """
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    # The `by_alias=True` is important to get the hyphenated keys.
    data_dict = data.dict(by_alias=True)
    
    # DO NOT MODIFY: turn the dict into a Pandas DataFrame.
    input_df = pd.DataFrame([data_dict])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Use the process_data function to prepare the data for inference
    data_processed, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None, # No label column in inference data
        training=False,
        encoder=encoder,
        lb=lb,
    )
    
    # Run the inference to get a prediction (e.g., [0] or [1])
    prediction_raw = inference(model, data_processed)
    
    # Convert the raw prediction back to the original label (e.g., "<=50K")
    prediction_label = lb.inverse_transform(prediction_raw)[0]
    
    return {"prediction": prediction_label}