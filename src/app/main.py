import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import mlflow
from pydantic import BaseModel, conint, confloat
import pandas as pd
import json
import uvicorn


# Definir a URI de Tracking do MLflow logo no início
#mlflow.set_tracking_uri('http://localhost:5000')

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)


# Define the inputs expected in the request body as JSON
class Request(BaseModel):
    """
    Request model for the API, defining the input structure.

    Attributes:
        LIMIT_BAL (float): Credit limit.
        SEX (int): Gender.
        EDUCATION (int): Education level.
        MARRIAGE (int): Marital status.
        AGE (int): Age of the individual.
        PAY_0 to PAY_6 (int): History of past payments.
        BILL_AMT1 to BILL_AMT6 (float): Bill statement amount.
        PAY_AMT1 to PAY_AMT6 (float): Payment amount.
    """
    LIMIT_BAL: confloat(ge=0) = 20000.0
    SEX: conint(ge=1, le=2) = 2
    EDUCATION: conint(ge=1, le=4) = 2
    MARRIAGE: conint(ge=0, le=3) = 1
    AGE: conint(ge=18) = 24
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: confloat(ge=0) = 0.0
    BILL_AMT2: confloat(ge=0) = 0.0
    BILL_AMT3: confloat(ge=0) = 0.0
    BILL_AMT4: confloat(ge=0) = 0.0
    BILL_AMT5: confloat(ge=0) = 0.0
    BILL_AMT6: confloat(ge=0) = 0.0
    PAY_AMT1: confloat(ge=0) = 0.0
    PAY_AMT2: confloat(ge=0) = 0.0
    PAY_AMT3: confloat(ge=0) = 0.0
    PAY_AMT4: confloat(ge=0) = 0.0
    PAY_AMT5: confloat(ge=0) = 0.0
    PAY_AMT6: confloat(ge=0) = 0.0

# Create a FastAPI application
app = fastapi.FastAPI()

# Add CORS middleware to allow all origins, methods, and headers for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri=model_uri)
    
    print(f"Loaded model {model_uri}")


@app.post("/predict")
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}

# Run the app on port 5003
uvicorn.run(app=app, port=config["service_port"], host="0.0.0.0")
