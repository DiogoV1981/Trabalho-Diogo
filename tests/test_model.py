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
        model_uri=f"models:/{model_name}@{model_version}"
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
        'LIMIT_BAL': 500000.0,
        'SEX': 1,
        'EDUCATION': 3,
        'MARRIAGE': 2,
        'AGE': 45,
        'PAY_0': 2,
        'PAY_2': 2,
        'PAY_3': 2,
        'PAY_4': 2,
        'PAY_5': 2,
        'PAY_6': 2,
        'BILL_AMT1': 50000.0,
        'BILL_AMT2': 50000.0,
        'BILL_AMT3': 50000.0,
        'BILL_AMT4': 50000.0,
        'BILL_AMT5': 50000.0,
        'BILL_AMT6': 50000.0,
        'PAY_AMT1': 5000.0,
        'PAY_AMT2': 5000.0,
        'PAY_AMT3': 5000.0,
        'PAY_AMT4': 5000.0,
        'PAY_AMT5': 5000.0,
        'PAY_AMT6': 5000.0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1  # Ajuste conforme necessário


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
