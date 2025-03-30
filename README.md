# Avaliação do módulo de Operacionalização de Machine Learning - Projecto Individual

## Rumos Bank going live

The Rumos Bank é um banco que tem perdido bastante dinheiro devido à quantidade de créditos que fornece e que não são pagos dentro do prazo devido. 

Depois do banco te contratar, como data scientist de topo, para ajudares a prever os clientes que não irão cumprir os prazos, os resultados exploratórios iniciais são bastante promissores!

Mas o banco está algo receoso, já que teve uma má experiência anterior com uma equipa de data scientists, em que a transição dos resultados iniciais exploratórios até de facto conseguirem ter algo em produção durou cerca de 6 meses, bem acima da estimativa inicial.

Por causa desta prévia má experiência, o banco desta vez quer ter garantias que a passagem dos resultados iniciais para produção é feita de forma mais eficiente. O objetivo é que a equipa de engenharia consegue colocar o vosso modelo em produção em dias em vez de meses!

## Avaliação

Os componentes que vão ser avaliados neste projecto são:

* `README.md` atualizado
* Todas as alterações que fazem são trackeadas num repositório do github
* Ambiente do projecto (conda.yaml) definido de forma adequada
* Runs feitas no notebook `rumos_bank_leading_prediction.ipynb` estão documentadas, reproduzíveis, guardadas e facilmente comparáveis
* Os modelos utilizados estão registados e versionados num Model Registry
* O melhor modelo está a ser servido num serviço - não precisa de UI
* O serviço tem testes
* O serviço está containerizado
* O container do serviço é built, testado e enviado para um container registry num pipeline de CICD

Garantam que tanto o repositório do github como o package no github estão ambos públicos!

### Data limite de entrega

TBD

Deve ser enviada, até à data limite de entrega, um link para o vosso github (tem de estar público). Podem enviar este link para o meu email `lopesg.miguel@gmail.com` ou slack.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Índice
1. Tecnologias Utilizadas
2. Pré-requisitos
3. Configuração do Ambiente
4. Como Executar o Projeto
5. Como Testar o Serviço
6. Detalhes do Pipeline CI/CD

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Tecnologias Utilizadas
   Este projeto foi desenvolvido utilizando as seguintes tecnologias:
   *  Python (>=3.12):Linguagem principal para desenvolvimento do modelo e serviço 
   *  FastAPI (0.112.2): Framework para construção do serviço web.
   *  MLflow (2.18.0): Gestão de experimentss e registro de modelos.
   *  Docker: Containerização do serviço e do MLflow Tracking Server.
   *  Pytest (8.3.4): Framework para testes automatizados.
   *  Conda: Gestão das dependências e ambiente.
   *  Uvicorn (0.32.1): Servidor para execução da aplicação FastAPI.


2. Pré-requisito
Antes de executar o projeto, verifique se você possui as seguintes ferramentas instaladas:
Docker e Docker Compose
Conda (versão mínima: 4.10.0)
Git (para clonar o repositório)

3. Configuração do Ambiente
    3.1 - Clone este repositório:
          git clone https://github.com/DiogoV1981/Trabalho-Diogo.git
          cd Trabalho-Diogo

   3.2 - Configure o ambiente Conda:
         conda env create -f conda.yaml
         conda activate OML2

   3.3 - O arquivo app.json contém as configurações principais para a integração com o MLflow e o modelo. As configurações incluem:
         - model_name: Nome do modelo registrado no MLflow que será utilizado para predições (random_forest_v2).
         - model_version: Versão específica do modelo que será carregada do MLflow (neste caso, 1).
         - tracking_base_url: URL base do MLflow Tracking Server (normalmente http://localhost para ambiente local).
         - tracking_port: Porta configurada para o MLflow Tracking Server (a padrão é 5000).
         - service_port: Porta na qual o serviço FastAPI será disponibilizado (a padrão é 5003)

   3.4 - Certifique-se de que todas as dependências foram instaladas:
         conda list
   
4. Como Executar o Projeto
   a) Inicie os serviços com Docker Compose:
       O arquivo docker-compose.yaml gerencia o MLflow Tracking Server e o serviço FastAPI. Execute o comando:
       docker compose up -d

   b) Verifique os serviços:
      MLflow Tracking Server estará disponível em: http://localhost:5000 # Registo experiments de Machine Learning e acessar os modelos registrados.
      FastAPI Serviço (documentação Swagger): http://localhost:5003/docs # Documentação interativa da API para testar endpoints e entender os resultados.
      FastAPI Serviço (documentação ReDoc): http://localhost:5003/redoc  # Documentação visual que detalha os endpoints da API.

   c) Realize previsões: Use o endpoint /predict para enviar uma requisição POST e obter previsões:
      curl -X 'POST' \
  'http://localhost:5003/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "LIMIT_BAL": 20000.0,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 0.0,
    "BILL_AMT2": 0.0,
    "BILL_AMT3": 0.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 0.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
  }'

4.1 Componentes do Código
  FastAPI Serviço:
  O arquivo "src/app/main.py" implementa o serviço FastAPI. Os principais pontos incluem:
   - Configuração do MLflow: A API conecta ao MLflow Tracking Server para carregar o modelo correto. O modelo é carregado durante o evento de inicialização (`startup_event`).
   - Endpoint `/predict`: Recebe um JSON com dados de entrada, realiza o pré-processamento (escalonamento com MinMaxScaler) e usa o modelo carregado para gerar predições.
   - Escalonamento: Normaliza os dados contínuos para corresponder ao formato esperado pelo modelo. A Variável categóricas, como "SEX", não é escalonada  [0 , 1]. 

5. Como Testar o Serviço:
   a) Testes do Serviço através do comando pytest tests/test_service.py:
    * Verifica se o endpoint `/predict` funciona corretamente e retorna uma predição.
    * Valida que a API responde com código de status 200 para entradas válidas.
    * Testa a presença e o formato da chave `"prediction"` na resposta.
   
   b) Testes do Modelo através do comando pytest tests/test_model.py:
    * Confirma que o modelo registrado no MLflow pode ser carregado corretamente.
    * Verifica a consistência das predições para diferentes cenários (valores normalizados e extremos).
    * Garante que o formato de saída do modelo seja compatível (ex.: apenas uma predição por entrada).
  
6. Detalhes do Pipeline CI/CD
   Este projeto implementa um pipeline CI/CD com GitHub Actions. Aqui está o fluxo do pipeline (configuração do arquivo: .github/workflows/cicd.yml):
   a)  Build do Ambiente: Cria o ambiente com base no conda.yaml para rodar os testes.
   b)  Testes Automatizados: Executa todos os testes implementados no serviço e no modelo.
   c)  Publicação de Imagens: Cria e publica as imagens Docker do serviço no GitHub Container Registry.
