from utils import graphs as g
from utils.load_config import carregar_configuracao
from training import training

import numpy as np
import os

import pandas as pd

dados = {
    'Produto': ['Produto A', 'Produto B', 'Produto C', 'Produto D'],
    'Vendas': [200, 150, 300, 250]
}

# Criando o DataFrame
df = pd.DataFrame(dados)

grafico1 = g.Grafico(df['Produto'], df['Vendas'], df)

X_numeric = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

y_numeric = np.array([0, 1, 0])

path_file = os.path.join(os.path.dirname(__file__), 'resources')
path_config = os.path.join(path_file, 'config.yml')

config = carregar_configuracao(path_config)

treino1 = training.TrainModel(config)

X_train, X_test, y_train, y_test = training.train_test_split(X_numeric, y_numeric)

print(X_train)
print(X_test)
print(y_train)
print(y_test)


from sklearn.datasets import load_iris

# Carregar os dados de exemplo (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

treino1.training_models(X, y)