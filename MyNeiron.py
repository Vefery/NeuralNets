import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Загружаем данные FashionMNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразуем изображения в одномерные векторы
x_train = x_train.reshape(-1, 28*28) / 255.0  # Нормализуем значения пикселей
x_test = x_test.reshape(-1, 28*28) / 255.0

# Преобразуем метки в one-hot кодировку
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = x_train, x_test, y_train, y_test

# Параметры сети
input_size = X_train.shape[1]
hidden_size = 128  # Можно увеличить размер скрытого слоя для лучшего качества
output_size = 10  # 10 классов для FashionMNIST
learning_rate = 0.01
epochs = 1000

# Инициализация весов
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Функции активации
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

# Обучение
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    # Прямой проход
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Вычисление потерь
    loss = cross_entropy_loss(y_train_one_hot, A2)
    train_losses.append(loss)

    # Обратный проход
    dZ2 = A2 - y_train_one_hot
    dW2 = np.dot(A1.T, dZ2) / X_train.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X_train.shape[0]

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = np.dot(X_train.T, dZ1) / X_train.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X_train.shape[0]

    # Обновление весов
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Метрика точности
    if epoch % 100 == 0:
        pred_train = np.argmax(A2, axis=1)
        accuracy = accuracy_score(y_train, pred_train)
        train_accuracies.append(accuracy)

# Вывод кривых обучения
plt.figure(figsize=(12, 6))

# Потери
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Точность
plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
