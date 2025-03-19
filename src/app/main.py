import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import mlflow
from pydantic import BaseModel, conint, confloat
import pandas as pd
import json
import uvicorn
import os


# Definir a URI de Tracking do MLflow logo no início
#mlflow.set_tracking_uri('http://localhost:5000')

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)


# Define the inputs expected in the request body as JSON
class RequestModel(BaseModel):
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
    try:
        tracking_uri = os.getenv("TRACKING_URI")  # Obtém o valor do ambiente
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Tracking URI configurado: {tracking_uri}")

        # O URI do modelo usando variáveis de ambiente
        model_uri = f"models:/{os.getenv('MODEL_NAME')}/{os.getenv('MODEL_VERSION')}"
        print(f"Tentando carregar o modelo: {model_uri}")

        # Carrega o modelo do MLflow
        app.model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print(f"Modelo {model_uri} carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
#@app.on_event("startup")
#async def startup_event():
 #   """
  #  Ações ao iniciar a aplicação.
   # """
    #mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Carregar o modelo registrado
    #model_uri = f"models:/{config['model_name']}/{config['model_version']}"
    #app.model = mlflow.pyfunc.load_model(model_uri=model_uri)
    #print(f"Loaded model {model_uri}")


@app.post("/predict")
async def predict(input: RequestModel): 
    """
    Endpoint de predição.
    """
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})
    prediction = app.model.predict(input_df)
    return {"prediction": prediction.tolist()[0]}

# Run the app on port 5003
if __name__ == "__main__":
    uvicorn.run(app=app, port=config["service_port"], host="0.0.0.0")
