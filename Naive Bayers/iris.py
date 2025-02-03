import random
random.seed(42)  # Define a semente

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carregar os dados
data = pd.read_csv('iris.csv', header=0)
data = data.dropna(axis='rows')  # Remove NaN

# Armazena os nomes das classes
classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)
print("Número de linhas e colunas na matriz de atributos:", data.shape)

attributes = list(data.columns)
print(data.head(10))  # Corrigido

# Converter para NumPy
data = data.to_numpy()
nrow, ncol = data.shape
y = data[:, -1]
x = data[:, 0:ncol-1]

# Selecionando os conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
p = 0.7  # Fração de elementos no conjunto de treinamento
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, random_state=42)

# Função para calcular a densidade de probabilidade conjunta
def likelihood(y, Z):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    prob = 1
    for j in range(Z.shape[1]):
        m = np.mean(Z[:, j])
        s = np.std(Z[:, j])
        if s == 0:  # Prevenção de erro de divisão por zero
            s = 1e-6
        prob *= gaussian(y[j], m, s)
    return prob

# Criar um DataFrame para armazenar probabilidades
P = pd.DataFrame(data=np.zeros((x_test.shape[0], len(classes))), columns=classes)

# Cálculo da estimativa para cada classe
for i in range(len(classes)):
    elements = np.where(y_train == classes[i])[0]  
    Z = x_train[elements, :]  

    for j in range(x_test.shape[0]):
        x_sample = x_test[j, :]
        pj = likelihood(x_sample, Z)  
        P.loc[j, classes[i]] = pj * len(elements) / x_train.shape[0]  

# Exibir probabilidades
print(P.head(10))

# Exibir imagem
img = mpimg.imread('iris_type.jpg')  
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.imshow(img)
plt.show()

# Calcula a acurácia
from sklearn.metrics import accuracy_score

y_pred = []
for i in range(P.shape[0]):
    c = np.argmax(P.iloc[i].values)  
    y_pred.append(P.columns[c])

y_pred = np.array(y_pred)

# Calcular a acurácia
score = accuracy_score(y_pred, y_test)
print('Accuracy:', score)

# Matriz de confusão
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.array(y_test, dtype=str), np.array(y_pred, dtype=str))
print("Confusion Matrix:")
print(cm)

# Classificação usando Naive Bayes do Scikit-learn
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('Accuracy utilizando GaussianNB:', score)

# Classificação assumindo distribuição Bernoulli
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('Accuracy utilizando BernoulliNB:', score)

# Exibir previsões em DataFrame formatado
df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df.head(10))  
