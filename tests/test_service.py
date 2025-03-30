import pytest
import requests

BASE_URL = "http://localhost:5003"  

def test_predict():
    """
    Testa o endpoint /predict com dados de entrada válidos.
    Deve retornar uma previsão na resposta.
    """
    response = requests.post(f"{BASE_URL}/predict", json={
        'LIMIT_BAL': 20000.0,
        'SEX': 1,
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
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))
    assert response.json()["prediction"] == 0  

if __name__ == "__main__":
    pytest.main()