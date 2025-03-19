import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, HTTPException
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
    """
    Evento de inicialização da aplicação. Configura o MLflow e carrega o modelo.
    """
    try:
        # Configurar a URI do MLflow Tracking Server a partir da variável de ambiente
        tracking_uri = os.getenv("TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Tracking URI configurado: {tracking_uri}")

        # Construir a URI do modelo a partir das configurações
        model_uri = f"models:/{config['model_name']}/{config['model_version']}"
        print(f"Tentando carregar o modelo: {model_uri}")

        # Carregar o modelo do MLflow
        app.model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print(f"Modelo {model_uri} carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        app.model = None
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
    Endpoint para realizar predições usando o modelo carregado.
    """
    if not hasattr(app, "model") or app.model is None:
        raise HTTPException(status_code=500, detail="Modelo não está disponível no momento.")

    try:
        # Converter os dados recebidos para um DataFrame
        input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})
        
        # Fazer a predição
        prediction = app.model.predict(input_df)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante a predição: {e}")

# Run the app on port 5003
if __name__ == "__main__":
    uvicorn.run(app=app, port=5003, host="0.0.0.0")
