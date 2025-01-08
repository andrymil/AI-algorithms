import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from MLP import MLP, sigmoid, sigmoid_derivative, relu, relu_derivative, cross_entropy_loss, cross_entropy_loss_derivative
from concurrent.futures import ProcessPoolExecutor
import time
from itertools import product
import matplotlib.pyplot as plt

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




# Track accuracy over epochs

"""
ACCURACY OVER EPOCHS
"""

def track_accuracy_over_epochs(sample_epochs=np.linspace(1, 100, 10, dtype=int), learning_rate=0.01, batch_size=8, layers=[64, 128, 64, 10]):
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    if isinstance(sample_epochs, (int, np.integer)):
        sample_epochs = np.array([sample_epochs])
    
    #print("\n\nEpochs: ", sample_epochs)

    for epoch in sample_epochs:
        mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
        mlp.fit(X_train, y_train, epoch, batch_size=batch_size)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
        test_accuracy = accuracy_score(np.argmax(y_test, axis=1), mlp.predict(X_test))
        
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        #print(f"Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return sample_epochs, train_accuracies, val_accuracies, test_accuracies

def plot_accuracy_over_epochs(sample_epochs, train_accuracies, val_accuracies, test_accuracies):
    plt.plot(sample_epochs, train_accuracies, label='Train Accuracy')
    plt.plot(sample_epochs, val_accuracies, label='Validation Accuracy')
    plt.plot(sample_epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()
    
"""
ACCURACY OVER LEARNING RATE
"""

def track_accuracy_over_learning_rate(sample_learning_rates=np.linspace(0.001, 0.01, 10), epochs=100, batch_size=8, layers=[64, 128, 64, 10], round_digits=4):
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    sample_learning_rates = np.round(sample_learning_rates, round_digits)
    if isinstance(sample_learning_rates, float):
        sample_learning_rates = np.array([sample_learning_rates])
    #print("\n\nLearning Rates: ", sample_learning_rates)
    
    for learning_rate in sample_learning_rates:
        mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
        mlp.fit(X_train, y_train, epochs, batch_size=batch_size)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
        test_accuracy = accuracy_score(np.argmax(y_test, axis=1), mlp.predict(X_test))
        
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        #print(f"Learning Rate {learning_rate}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return sample_learning_rates, train_accuracies, val_accuracies, test_accuracies

def plot_accuracy_over_learning_rate(sample_learning_rates, train_accuracies, val_accuracies, test_accuracies):
    plt.plot(sample_learning_rates, train_accuracies, label='Train Accuracy')
    plt.plot(sample_learning_rates, val_accuracies, label='Validation Accuracy')
    plt.plot(sample_learning_rates, test_accuracies, label='Test Accuracy')
    plt.xlabel('Learning Rate')
    plt.xticks(np.linspace(min(sample_learning_rates), max(sample_learning_rates), num=20))
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.legend()
    plt.show()
    

"""
ACCURACY OVER BATCH SIZE
"""

def track_accuracy_over_batch_size(sample_batch_sizes=np.linspace(1, 64, 10, dtype=int), epochs=100, learning_rate=0.01, layers=[64, 128, 64, 10]):
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    if isinstance(sample_batch_sizes, (int, np.integer)):
        sample_batch_sizes = np.array([sample_batch_sizes])
    
    #print("\n\nBatch Sizes: ", sample_batch_sizes)
    
    for batch_size in sample_batch_sizes:
        mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
        mlp.fit(X_train, y_train, epochs, batch_size=batch_size)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
        test_accuracy = accuracy_score(np.argmax(y_test, axis=1), mlp.predict(X_test))
        
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        #print(f"Batch Size {batch_size}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return sample_batch_sizes, train_accuracies, val_accuracies, test_accuracies

def plot_accuracy_over_batch_size(sample_batch_sizes, train_accuracies, val_accuracies, test_accuracies):
    plt.plot(sample_batch_sizes, train_accuracies, label='Train Accuracy')
    plt.plot(sample_batch_sizes, val_accuracies, label='Validation Accuracy')
    plt.plot(sample_batch_sizes, test_accuracies, label='Test Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Batch Size')
    plt.legend()
    plt.show()

"""
ACCURACY OVER LAYERS
"""
    
def track_accuracy_over_layers(sample_layers=[[64, 128, 64, 10], [64, 128, 10], [64, 10]], epochs=100, learning_rate=0.01, batch_size=8):
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    #print("\n\nLayers: ", sample_layers)
    
    if isinstance(sample_layers[0], (int, np.integer)):
        sample_layers = [sample_layers]
    
    for layers in sample_layers:
        mlp = MLP(layers=layers, activation=(relu, relu_derivative), loss=(cross_entropy_loss, cross_entropy_loss_derivative), learning_rate=learning_rate)
        mlp.fit(X_train, y_train, epochs, batch_size=batch_size)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), val_pred)
        test_accuracy = accuracy_score(np.argmax(y_test, axis=1), mlp.predict(X_test))
        
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        #print(f"Layers {layers}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return sample_layers, train_accuracies, val_accuracies, test_accuracies

def plot_accuracy_over_layers(sample_layers, train_accuracies, val_accuracies, test_accuracies):
    sample_layers_str = [str(layers[1:-1]) for layers in sample_layers]
    plt.plot(sample_layers_str, train_accuracies, label='Train Accuracy')
    plt.plot(sample_layers_str, val_accuracies, label='Validation Accuracy')
    plt.plot(sample_layers_str, test_accuracies, label='Test Accuracy')
    plt.xlabel('Layers')
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Layers')
    plt.legend()
    plt.show()


## Testing
#plot_accuracy_over_epochs(*track_accuracy_over_epochs(1, 100, 10))

def parallel_track_accuracy_over_epochs(_):
    return track_accuracy_over_epochs(1, 100, 10)

def parallel_track_accuracy_over_epochs_with_boxplot(runs=3):
    start_time = time.time()
    
    m_train_accuracies = []
    m_val_accuracies = []
    m_test_accuracies = []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_track_accuracy_over_epochs, range(0, runs-1)))
        
    for sample_epochs, train_accuracies, val_accuracies, test_accuracies in results:
        m_val_accuracies.append(val_accuracies)
    
    m_val_accuracies = np.array(m_val_accuracies).T.tolist()
            
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    plt.boxplot(m_val_accuracies, positions=np.linspace(1, 100, 10, dtype=int), widths=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Epochs')
    plt.show()
    
def parallel_track_accuracy_over_learning_rate_fix(sample):
    return track_accuracy_over_learning_rate(sample)

def parallel_track_accuracy_over_learning_rate(m_sample_learning_rates=np.linspace(0.0002, 0.3, 50)):
    
    
    m_sample_learning_rates = np.round(m_sample_learning_rates, 5)
    #samples = [(m_sample_learning_rates[i], m_sample_learning_rates[i + 1]) for i in range(0, len(m_sample_learning_rates), 2)]
    m_train_accuracies = []
    m_val_accuracies = []
    m_test_accuracies = []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_track_accuracy_over_learning_rate_fix, m_sample_learning_rates))
        
    for sample_learning_rates, train_accuracies, val_accuracies, test_accuracies in results:
        for train_accuracy in train_accuracies:
            m_train_accuracies.append(train_accuracy)
        for val_accuracy in val_accuracies:
            m_val_accuracies.append(val_accuracy)
        for test_accuracy in test_accuracies:
            m_test_accuracies.append(test_accuracy)
        
    return m_sample_learning_rates, m_train_accuracies, m_val_accuracies, m_test_accuracies

def parallel_track_accuracy_over_epochs_fix(sample):
    return track_accuracy_over_epochs(sample)

def parallel_track_accuracy_over_epochs(m_sample_epochs=np.linspace(1, 100, 10, dtype=int)):
    
    
    m_train_accuracies = []
    m_val_accuracies = []
    m_test_accuracies = []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(track_accuracy_over_epochs, m_sample_epochs))
        
    for sample_learning_rates, train_accuracies, val_accuracies, test_accuracies in results:
        for train_accuracy in train_accuracies:
            m_train_accuracies.append(train_accuracy)
        for val_accuracy in val_accuracies:
            m_val_accuracies.append(val_accuracy)
        for test_accuracy in test_accuracies:
            m_test_accuracies.append(test_accuracy)
        
    return m_sample_epochs, m_train_accuracies, m_val_accuracies, m_test_accuracies

def parallel_track_accuracy_over_batch_size_fix(sample):
    return track_accuracy_over_batch_size(sample)

def parallel_track_accuracy_over_batch_size(m_sample_batch_sizes=np.linspace(1, 64, 10, dtype=int)):
    
    
    m_train_accuracies = []
    m_val_accuracies = []
    m_test_accuracies = []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_track_accuracy_over_batch_size_fix, m_sample_batch_sizes))
        
    for sample_batch_sizes, train_accuracies, val_accuracies, test_accuracies in results:
        for train_accuracy in train_accuracies:
            m_train_accuracies.append(train_accuracy)
        for val_accuracy in val_accuracies:
            m_val_accuracies.append(val_accuracy)
        for test_accuracy in test_accuracies:
            m_test_accuracies.append(test_accuracy)
        
    return m_sample_batch_sizes, m_train_accuracies, m_val_accuracies, m_test_accuracies

def parallel_track_accuracy_over_layers_fix(sample):
    return track_accuracy_over_layers(sample)

def parallel_track_accuracy_over_layers(m_sample_layers=[[64, 128, 64, 10], [64, 128, 10], [64, 10]]):
    
    m_train_accuracies = []
    m_val_accuracies = []
    m_test_accuracies = []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_track_accuracy_over_layers_fix, m_sample_layers))
        
    for sample_layers, train_accuracies, val_accuracies, test_accuracies in results:
        for train_accuracy in train_accuracies:
            m_train_accuracies.append(train_accuracy)
        for val_accuracy in val_accuracies:
            m_val_accuracies.append(val_accuracy)
        for test_accuracy in test_accuracies:
            m_test_accuracies.append(test_accuracy)
        
    return m_sample_layers, m_train_accuracies, m_val_accuracies, m_test_accuracies

def generate_layer_configurations(num_layers, min_units, max_units):
    
    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1")
    
    hidden_layer_combinations = product(range(min_units, max_units + 1), repeat=num_layers)
    
    layer_configurations = [[64] + list(hidden_layers) + [10] for hidden_layers in hidden_layer_combinations]
    
    return layer_configurations

def parallel_track_accuracy_over_layers_median(runs=10, m_sample_layers=[[64, 128, 64, 10], [64, 128, 10], [64, 10]]):
    
    all_train_accuracies = []
    all_val_accuracies = []
    all_test_accuracies = []
    
    for _ in range(runs):
        sample_layers, train_accuracies, val_accuracies, test_accuracies = parallel_track_accuracy_over_layers(m_sample_layers)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)
        all_test_accuracies.append(test_accuracies)
        
    median_train_accuracies = np.median(all_train_accuracies, axis=0)
    median_val_accuracies = np.median(all_val_accuracies, axis=0)
    median_test_accuracies = np.median(all_test_accuracies, axis=0)
    
    plot_accuracy_over_layers(sample_layers, median_train_accuracies, median_val_accuracies, median_test_accuracies)
        

if __name__ == '__main__':
    plot_accuracy_over_epochs(*track_accuracy_over_epochs(sample_epochs=np.linspace(1, 100, 10, dtype=int)))
    plot_accuracy_over_epochs(*parallel_track_accuracy_over_epochs(m_sample_epochs=np.linspace(1, 80, 40, dtype=int)))
    plot_accuracy_over_learning_rate(*track_accuracy_over_learning_rate(sample_learning_rates=np.linspace(0.001, 0.2, 100)))
    plot_accuracy_over_learning_rate(*parallel_track_accuracy_over_learning_rate(m_sample_learning_rates=np.linspace(0.0001, 0.3, 100)))
    #plot_accuracy_over_batch_size(*track_accuracy_over_batch_size(sample_batch_sizes=np.linspace(1, 64, 10, dtype=int)))
    # plot_accuracy_over_layers(*track_accuracy_over_layers([[64, 128, 64, 10], [64, 128, 10], [64, 1, 10]]))
    
    
    
    
    
    
    
    
    
    
    # runs = 10
    # all_train_accuracies = []
    # all_val_accuracies = []
    # all_test_accuracies = []

    # for _ in range(runs):
    #     sample_learning_rates, train_accuracies, val_accuracies, test_accuracies = parallel_track_accuracy_over_epochs(m_sample_epochs=np.linspace(1, 100, 100, dtype=int))
    #     all_train_accuracies.append(train_accuracies)
    #     all_val_accuracies.append(val_accuracies)
    #     all_test_accuracies.append(test_accuracies)

    # median_train_accuracies = np.median(all_train_accuracies, axis=0)
    # median_val_accuracies = np.median(all_val_accuracies, axis=0)
    # median_test_accuracies = np.median(all_test_accuracies, axis=0)

    # plot_accuracy_over_epochs(sample_learning_rates, median_train_accuracies, median_val_accuracies, median_test_accuracies)
    
    
    
    #plot_accuracy_over_batch_size(*track_accuracy_over_batch_size(sample_batch_sizes=np.linspace(1, 64, 10, dtype=int)))
    # plot_accuracy_over_batch_size(*parallel_track_accuracy_over_batch_size(m_sample_batch_sizes=np.linspace(1, 64, 10, dtype=int)))
    
    # runs = 15
    # all_train_accuracies = []
    # all_val_accuracies = []
    # all_test_accuracies = []
    
    # for _ in range(runs):
    #     sample_batch_sizes, train_accuracies, val_accuracies, test_accuracies = parallel_track_accuracy_over_batch_size(m_sample_batch_sizes=np.linspace(1, 64, 16, dtype=int))
    #     all_train_accuracies.append(train_accuracies)
    #     all_val_accuracies.append(val_accuracies)
    #     all_test_accuracies.append(test_accuracies)
        
    # median_train_accuracies = np.median(all_train_accuracies, axis=0)
    # median_val_accuracies = np.median(all_val_accuracies, axis=0)
    # median_test_accuracies = np.median(all_test_accuracies, axis=0)
    
    # plot_accuracy_over_batch_size(sample_batch_sizes, median_train_accuracies, median_val_accuracies, median_test_accuracies)
    

    
    # plot_accuracy_over_layers(*track_accuracy_over_layers([[64, 128, 64, 10], [64, 128, 10], [64, 1, 10]]))
    
    #parallel_track_accuracy_over_layers_median(runs=5, m_sample_layers=[[64, 512, 10], [64, 8, 64, 10], [64, 16, 32, 10], [64, 16, 32, 10], [64, 32, 16, 10], [64, 64, 8, 10]])
    #parallel_track_accuracy_over_layers_median(runs=3, m_sample_layers=generate_layer_configurations(2, 32, 256)[::32])
    #parallel_track_accuracy_over_layers_median(runs=5, m_sample_layers=sorted(generate_layer_configurations(2, 4, 16), key=lambda x: x[0] * x[1]))
    
    #parallel_track_accuracy_over_epochs_with_boxplot(5)

