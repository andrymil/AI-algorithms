import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true

class MLP:
    def __init__(self, layers, activation, loss, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation_function = activation[0]
        self.activation_derivative = activation[1]
        self.loss_function = loss[0]
        self.loss_derivative = loss[1]

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, X):
        activations = [X]

        for W, b in zip(self.weights, self.biases):
            input_to_layer = np.dot(activations[-1], W) + b

            if W is self.weights[-1]:
                activation = softmax(input_to_layer)
            else:
                activation = self.activation_function(input_to_layer)

            activations.append(activation)

        return activations

    def backward(self, activations, y_true):
        gradients_w = []
        gradients_b = []

        loss_grad = self.loss_derivative(y_true, activations[-1])
        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(activations[i].T, loss_grad) / y_true.shape[0]
            grad_b = np.mean(loss_grad, axis=0, keepdims=True)

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if i > 0:
                loss_grad = np.dot(loss_grad, self.weights[i].T) * self.activation_derivative(activations[i])

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def fit(self, X, y, epochs=100, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                activations = self.forward(X_batch)
                gradients_w, gradients_b = self.backward(activations, y_batch)
                self.update_weights(gradients_w, gradients_b)

            activations = self.forward(X)
            loss = self.loss_function(y, activations[-1])
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

# Load MNIST dataset
digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train MLP model
mlp = MLP(layers=[64, 128, 64, 10], activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=0.01)
mlp.fit(X_train, y_train, epochs=100, batch_size=8)

# Evaluate the model
y_pred = mlp.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
