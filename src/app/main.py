import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import mlflow
from pydantic import BaseModel, conint, confloat
import pandas as pd
import json
import uvicorn
import os


# Carregar as configurações da aplicação
with open('./config/app.json') as f:
    config = json.load(f)

# Parâmetros para escalonamento MinMaxScaler - > identificado o min e max do dataset
scaling_params = {
    "LIMIT_BAL": {"min": 10000.0, "max": 1000000.0},
    #"SEX": {"min": 1.0, "max": 2.0}, # não quero escalar, valores possíveis 0 ou 1
    "EDUCATION": {"min": 0.0, "max": 6.0},
    "MARRIAGE": {"min": 0.0, "max": 3.0},
    "AGE": {"min": 21.0, "max": 79.0},
    "PAY_0": {"min": -2.0, "max": 8.0},
    "PAY_2": {"min": -2.0, "max": 8.0},
    "PAY_3": {"min": -2.0, "max": 8.0},
    "PAY_4": {"min": -2.0, "max": 8.0},
    "PAY_5": {"min": -2.0, "max": 8.0},
    "PAY_6": {"min": -2.0, "max": 8.0},
    "BILL_AMT1": {"min": -165580.0, "max": 964511.0},
    "BILL_AMT2": {"min": -69777.0, "max": 983931.0},
    "BILL_AMT3": {"min": -157264.0, "max": 1664089.0},
    "BILL_AMT4": {"min": -170000.0, "max": 891586.0},
    "BILL_AMT5": {"min": -81334.0, "max": 927171.0},
    "BILL_AMT6": {"min": -209051.0, "max": 961664.0},
    "PAY_AMT1": {"min": 0.0, "max": 873552.0},
    "PAY_AMT2": {"min": 0.0, "max": 1684259.0},
    "PAY_AMT3": {"min": 0.0, "max": 896040.0},
    "PAY_AMT4": {"min": 0.0, "max": 621000.0},
    "PAY_AMT5": {"min": 0.0, "max": 426529.0},
    "PAY_AMT6": {"min": 0.0, "max": 527143.0}
}

# Variáveis categóricas que não devem ser escalonadas
categorical_features = ["SEX"]


# Definir os inputs esperados no corpo da requisição como JSON
class RequestModel(BaseModel):
    LIMIT_BAL: confloat(ge=10000.0, le=1000000.0) = 20000.0
    SEX: conint(ge=0, le=1) = 1
    EDUCATION: conint(ge=0, le=6) = 2  # Aceita valores de 0 a 6
    MARRIAGE: conint(ge=0, le=3) = 2
    AGE: conint(ge=21, le=79) = 24  # Baseado nos valores do dataset
    PAY_0: conint(ge=-2, le=8) = 0
    PAY_2: conint(ge=-2, le=8) = 0
    PAY_3: conint(ge=-2, le=8) = 0
    PAY_4: conint(ge=-2, le=8) = 0
    PAY_5: conint(ge=-2, le=8) = 0
    PAY_6: conint(ge=-2, le=8) = 0
    BILL_AMT1: confloat(ge=-165580.0, le=964511.0) = 0.0  # Aceita valores negativos
    BILL_AMT2: confloat(ge=-69777.0, le=983931.0) = 0.0
    BILL_AMT3: confloat(ge=-157264.0, le=1664089.0) = 0.0
    BILL_AMT4: confloat(ge=-170000.0, le=891586.0) = 0.0
    BILL_AMT5: confloat(ge=-81334.0, le=927171.0) = 0.0
    BILL_AMT6: confloat(ge=-209051.0, le=961664.0) = 0.0
    PAY_AMT1: confloat(ge=0, le=873552.0) = 0.0
    PAY_AMT2: confloat(ge=0, le=1684259.0) = 0.0
    PAY_AMT3: confloat(ge=0, le=896040.0) = 0.0
    PAY_AMT4: confloat(ge=0, le=621000.0) = 0.0
    PAY_AMT5: confloat(ge=0, le=426529.0) = 0.0
    PAY_AMT6: confloat(ge=0, le=527143.0) = 0.0


# Criar a aplicação FastAPI
app = fastapi.FastAPI()

# Adicionar middleware CORS para permitir todas as origens, métodos e cabeçalhos
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


def minmax_scale(value, feature_name, scaling_params):
    """
    Escalona um valor usando os parâmetros min e max do MinMaxScaler.
    """
    min_val = scaling_params[feature_name]["min"]
    max_val = scaling_params[feature_name]["max"]
    return (value - min_val) / (max_val - min_val)


@app.post("/predict")
async def predict(input: RequestModel):
    """
    Endpoint para realizar predições usando o modelo carregado.
    """
    if not hasattr(app, "model") or app.model is None:
        raise HTTPException(status_code=500, detail="Modelo não está disponível no momento.")

    try:
        # Aplicar escalonamento apenas às variáveis contínuas
        input_data = {}
        for key, value in input.dict().items():
            if key in categorical_features:
                input_data[key] = value  # Não aplicar escalonamento
            else:
                input_data[key] = minmax_scale(value, key, scaling_params)  # Escalonar variáveis contínuas

       # Logs detalhados
        print(f"Valores recebidos pelo usuário: {input.dict()}")
        print(f"Valores escalonados enviados ao modelo: {input_data}")

        # Converter os dados para um DataFrame
        input_df = pd.DataFrame.from_dict({k: [v] for k, v in input_data.items()})
        print(f"DataFrame final enviado ao modelo:\n{input_df}")     
          
        # Fazer a predição
        prediction = app.model.predict(input_df)
        print(f"Resultado da predição: {prediction}")       
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante a predição: {e}")


# Executar a aplicação na porta 5003
if __name__ == "__main__":
    uvicorn.run(app=app, port=5003, host="0.0.0.0")