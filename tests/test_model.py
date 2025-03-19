import json
import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@champion"#model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 20000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 24,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 0.0,
        'BILL_AMT2': 0.0,
        'BILL_AMT3': 0.0,
        'BILL_AMT4': 0.0,
        'BILL_AMT5': 0.0,
        'BILL_AMT6': 0.0,
        'PAY_AMT1': 0.0,
        'PAY_AMT2': 0.0,
        'PAY_AMT3': 0.0,
        'PAY_AMT4': 0.0,
        'PAY_AMT5': 0.0,
        'PAY_AMT6': 0.0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0  # Ajuste conforme necessário


def test_model_dir(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 0.02,          # Limite baixo (escalado)
        'SEX': 0.0,                 # Masculino (escalado)
        'EDUCATION': 0.33,          # Educação intermediária (escalado)
        'MARRIAGE': 0.66,           # Estado civil (escalado)
        'AGE': 0.06,                # Idade jovem (escalado)
        'PAY_0': 1,               # Atrasos frequentes no pagamento (escalado)
        'PAY_2': 1,               # Atraso em 2 meses (escalado)
        'PAY_3': 1,               # Atraso adicional (escalado)
        'PAY_4': 1,               # Pagamento recente em dia (escalado)
        'PAY_5': 1,               # Pagamento em dia (escalado)
        'PAY_6': 1,               # Pagamento em dia (escalado)
        'BILL_AMT1': 0.9,           # Alto saldo pendente
        'BILL_AMT2': 0.88,          # Saldo ligeiramente menor
        'BILL_AMT3': 0.87,          # Contas com valores consistentes
        'BILL_AMT4': 0.86,          # Continuação de faturas altas
        'BILL_AMT5': 0.75,          # Saldo alto
        'BILL_AMT6': 0.74,          # Consistente com valores elevados
        'PAY_AMT1': 0.0,           # Pagamento baixo (escalado)
        'PAY_AMT2': 0.0,           # Pagamento baixo
        'PAY_AMT3': 0.0,           # Pagamento baixo
        'PAY_AMT4': 0.0,           # Pagamento baixo
        'PAY_AMT5': 0.0,           # Pagamento baixo
        'PAY_AMT6': 0.0            # Pagamento baixo
    }])
    prediction = model.predict(data=input)
    print("Predição retornada pelo modelo:", prediction)   
    assert prediction[0] == 1  # Validar classe 1


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 20000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 24,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 0.0,
        'BILL_AMT2': 0.0,
        'BILL_AMT3': 0.0,
        'BILL_AMT4': 0.0,
        'BILL_AMT5': 0.0,
        'BILL_AMT6': 0.0,
        'PAY_AMT1': 0.0,
        'PAY_AMT2': 0.0,
        'PAY_AMT3': 0.0,
        'PAY_AMT4': 0.0,
        'PAY_AMT5': 0.0,
        'PAY_AMT6': 0.0
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )

if __name__ == "__main__":
    pytest.main()
