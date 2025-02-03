import random
random.seed(42)  # Define a semente aleatória para garantir reprodutibilidade dos resultados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Biblioteca para exibir imagens

# 📌 Carregar os dados do conjunto de dados Iris
data = pd.read_csv('iris.csv', header=0)  # Lê o arquivo CSV contendo os dados da íris
data = data.dropna(axis='rows')  # Remove valores ausentes (NaN) para evitar erros

# 📌 Armazena os nomes das classes (espécies de flores)
classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)  # Obtém os nomes das classes únicas
print("Número de linhas e colunas na matriz de atributos:", data.shape)  # Exibe a dimensão do dataset

# 📌 Lista os nomes das colunas (atributos)
attributes = list(data.columns)
print(data.head(10))  # Exibe as primeiras 10 linhas do dataset para ver os dados

# 📌 Converter os dados para um array NumPy para facilitar o processamento
data = data.to_numpy()
nrow, ncol = data.shape  # Obtém o número de linhas e colunas
y = data[:, -1]  # Última coluna representa as classes das flores (rótulos)
x = data[:, 0:ncol-1]  # Demais colunas são os atributos

# 📌 Dividir os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
p = 0.7  # Define que 70% dos dados serão usados para treinamento e 30% são para teste
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, random_state=42)

# 📌 Definição de uma função para calcular a densidade de probabilidade conjunta usando distribuição Gaussiana
def likelihood(y, Z):
    def gaussian(x, mu, sig):
        """Função da distribuição Gaussiana (Normal)"""
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    prob = 1
    for j in range(Z.shape[1]):  # Percorre todas as colunas (atributos)
        m = np.mean(Z[:, j])  # Calcula a média dos valores do atributo
        s = np.std(Z[:, j])  # Calcula o desvio padrão do atributo
        if s == 0:  # Evita divisão por zero
            s = 1e-6
        prob *= gaussian(y[j], m, s)  # Multiplica as probabilidades individuais
    return prob

# 📌 Criar um DataFrame para armazenar probabilidades de cada classe
P = pd.DataFrame(data=np.zeros((x_test.shape[0], len(classes))), columns=classes)

# 📌 Cálculo da probabilidade de cada classe para cada amostra do conjunto de teste
for i in range(len(classes)):  # Para cada classe no conjunto de dados
    elements = np.where(y_train == classes[i])[0]  # Obtém os índices das amostras da classe atual
    Z = x_train[elements, :]  # Filtra os dados da classe correspondente

    for j in range(x_test.shape[0]):  # Para cada amostra do conjunto de teste
        x_sample = x_test[j, :]
        pj = likelihood(x_sample, Z)  # Calcula a probabilidade da amostra pertencer à classe
        P.loc[j, classes[i]] = pj * len(elements) / x_train.shape[0]  # Multiplica pela probabilidade a priori

# 📌 Exibir as probabilidades calculadas
print(P.head(10))

# 📌 Exibir uma imagem ilustrativa
img = mpimg.imread('iris_type.jpg')  
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.imshow(img)
plt.show()

# 📌 Calcula a acurácia do classificador manual
from sklearn.metrics import accuracy_score

y_pred = []
for i in range(P.shape[0]):  # Para cada amostra no conjunto de teste
    c = np.argmax(P.iloc[i].values)  # Obtém o índice da classe com maior probabilidade
    y_pred.append(P.columns[c])  # Armazena o nome da classe prevista

y_pred = np.array(y_pred)  # Converte a lista para array NumPy

# 📌 Calcular e exibir a acurácia do modelo
score = accuracy_score(y_pred, y_test)
print('Accuracy:', score)

# 📌 Exibir a matriz de confusão para avaliar os erros do modelo
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.array(y_test, dtype=str), np.array(y_pred, dtype=str))
print("Confusion Matrix:")
print(cm)

# 📌 Testes independentes usando bibliotecas do Scikit-Learn
# Classificação usando Naive Bayes Gaussiano (assume distribuição normal dos atributos)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)  # Treina o modelo
y_pred = model.predict(x_test)  # Faz previsões no conjunto de teste
score = accuracy_score(y_pred, y_test)  # Calcula a acurácia
print('Accuracy utilizando GaussianNB:', score)

# 📌 Teste usando Bernoulli Naive Bayes (assume atributos binários)
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(x_train, y_train)  # Treina o modelo
y_pred = model.predict(x_test)  # Faz previsões no conjunto de teste
score = accuracy_score(y_pred, y_test)  # Calcula a acurácia
print('Accuracy utilizando BernoulliNB:', score)

# 📌 Exibir previsões comparando valores reais e previstos
df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df.head(10))  
