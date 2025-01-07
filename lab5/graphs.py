import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from MLP import MLP, sigmoid, sigmoid_derivative, relu, relu_derivative, cross_entropy_loss, cross_entropy_loss_derivative
from concurrent.futures import ProcessPoolExecutor
import time


digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

"""
# Create and train MLP model
mlp = MLP(layers=[64, 128, 64, 10], activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=0.01)
mlp.fit(X_train, y_train, epochs=100, batch_size=8)

# Evaluate the model
y_pred = mlp.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)
"""


import matplotlib.pyplot as plt

# Track accuracy over epochs

def track_accuracy_over_epochs(start_epoch, end_epoch, num_samples, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10]):
    train_accuracies = []
    val_accuracies = []

    sample_epochs = np.linspace(start_epoch, end_epoch, num_samples, dtype=int)

    for epoch in sample_epochs:
        mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
        mlp.fit(X_train, y_train, epoch, batch_size=batch_size)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return sample_epochs, train_accuracies, val_accuracies

def plot_accuracy_over_epochs(sample_epochs, train_accuracies, val_accuracies):
    plt.plot(sample_epochs, train_accuracies, label='Train Accuracy')
    plt.plot(sample_epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()


def track_accuracy_over_epochs_with_boxplot(start_epoch, end_epoch, num_samples, num_runs=3, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10]):
    start_time = time.time()
    
    train_accuracies = []
    val_accuracies = []

    sample_epochs = np.linspace(start_epoch, end_epoch, num_samples, dtype=int)

    for epoch in sample_epochs:
        train_acc = []
        val_acc = []
        for _ in range(num_runs):
            mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
            mlp.fit(X_train, y_train, epoch, batch_size=batch_size)
            train_pred = mlp.predict(X_train)
            val_pred = mlp.predict(X_val)
            train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
            val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch}, Train Accuracy: {np.mean(train_acc):.4f}, Validation Accuracy: {np.mean(val_acc):.4f}")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # Plot the accuracies
    plt.boxplot(val_accuracies, positions=sample_epochs, widths=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.show()
    



def evaluate_learning_rates_boxplot(start, end, num_samples, num_runs=2, epochs=100, batch_size=8, layers=[64, 128, 64, 10]):
    train_accuracies = []
    val_accuracies = []
    accuracy = 0

    learning_rates = np.linspace(start, end, num_samples, dtype=float)
    learning_rates = np.round(learning_rates, 4)

    for learning_rate in learning_rates:
        train_acc = []
        val_acc = []
        for _ in range(num_runs):
            mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
            mlp.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
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
    plt.boxplot(val_accuracies, positions=learning_rates, widths=0.005)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Learning Rate')
    plt.show()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    
#track_accuracy_over_epochs(start_epoch=1, end_epoch=300, num_samples=20, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10])
#track_accuracy_over_epochs_with_boxplot(start_epoch=1, end_epoch=300, num_samples=20, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10])
#evaluate_learning_rates_boxplot(start=0.0001, end=0.11, num_samples=10, num_runs=2, epochs=100, batch_size=8, layers=[64, 128, 64, 10])

def track_accuracy_over_epochs_with_boxplot_parallel(start_epoch, end_epoch, num_samples, num_runs=3, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10]):
    start_time = time.time()
    
    def train_and_evaluate(epoch):
        train_acc = []
        val_acc = []
        for _ in range(num_runs):
            mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
            mlp.fit(X_train, y_train, epoch, batch_size=batch_size)
            train_pred = mlp.predict(X_train)
            val_pred = mlp.predict(X_val)
            train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
            val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
        return train_acc, val_acc

    sample_epochs = np.linspace(start_epoch, end_epoch, num_samples, dtype=int)
    train_accuracies = []
    val_accuracies = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(train_and_evaluate, sample_epochs))

    for train_acc, val_acc in results:
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Epoch {sample_epochs[len(train_accuracies)-1]}, Train Accuracy: {np.mean(train_acc):.4f}, Validation Accuracy: {np.mean(val_acc):.4f}")
    
        
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # Plot the accuracies
    plt.boxplot(val_accuracies, positions=sample_epochs, widths=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.show()

def track_accuracy_over_epochs_with_boxplot_parallel2(start_epoch, end_epoch, num_samples, num_runs=3, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10]):
    start_time = time.time()
    
    train_accuracies = []
    val_accuracies = []
    
    def test(_):
        return track_accuracy_over_epochs(start_epoch, end_epoch, num_samples, learning_rate, batch_size, layers)
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(test, range(1, num_runs+1)))
    
    for train_acc, val_acc in results:
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    
        
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # # Plot the accuracies
    # plt.boxplot(val_accuracies, positions=runs, widths=0.5)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Validation Accuracy vs Epochs')
    # plt.show()
    

    
    
# track_accuracy_over_epochs_with_boxplot(start_epoch=1, end_epoch=300, num_samples=20, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10])
# track_accuracy_over_epochs_with_boxplot_parallel(start_epoch=1, end_epoch=300, num_samples=20, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10])
#track_accuracy_over_epochs_with_boxplot_parallel2(start_epoch=1, end_epoch=300, num_samples=20, num_runs=3, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10])

def test(_):
    return track_accuracy_over_epochs(1, 100, 10, learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10])

def test_wrapper():
    # report a message
    print('Starting task...')
    start_time = time.time()
    
    train_accuracies = []
    val_accuracies = []
    # create the process pool
    with ProcessPoolExecutor() as exe:
        # perform calculations
        results = exe.map(test, range(0, 3))
        
    print("Results:")

    for sample_epochs, train_acc, val_acc in results:
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    test_wrapper()
        
    # report a message
    print('Done.')