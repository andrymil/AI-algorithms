import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from MLP import MLP, sigmoid, sigmoid_derivative, relu, relu_derivative, cross_entropy_loss, cross_entropy_loss_derivative


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
import matplotlib.pyplot as plt

# Track accuracy over epochs
"""
train_accuracies = []
val_accuracies = []

sample_value = np.linspace(1, 300, 20, dtype=int)

for epoch in sample_value:
    mlp = MLP(layers=[64, 128, 64, 10], activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=0.01)
    mlp.fit(X_train, y_train, epoch, batch_size=8)
    train_pred = mlp.predict(X_train)
    val_pred = mlp.predict(X_val)
    train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
    val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Plot the accuracies
plt.plot(sample_value, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.show()
print(f"Test Accuracy: {accuracy:.4f}")
"""

train_accuracies = []
val_accuracies = []

sample_value = np.linspace(0.001, 0.5, 20, dtype=float)

for learning_rate in sample_value:
    train_acc = []
    val_acc = []
    for _ in range(10):  # Run 10 times for each learning rate
        mlp = MLP(layers=[64, 128, 64, 10], activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
        mlp.fit(X_train, y_train, epochs=100, batch_size=8)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Learning rate {learning_rate}, Train Accuracy: {np.mean(train_acc):.5f}, Validation Accuracy: {np.mean(val_acc):.5f}")

# Plot the accuracies
plt.boxplot(val_accuracies, positions=sample_value, widths=0.005)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs Learning Rate')
plt.show()
print(f"Test Accuracy: {accuracy:.4f}")