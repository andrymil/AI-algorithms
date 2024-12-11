from ID3_algorithm import ID3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def load_and_preprocess_data(file_path, continuous_features, q=10):
    """Load and preprocess the data with a specific number of bins (q)."""
    data = pd.read_csv(file_path, sep=';')
    for feature in continuous_features:
        data[feature] = pd.qcut(data[feature], q=q, labels=False, duplicates='drop')
    X = data.drop(columns=['id', 'cardio'])
    y = data['cardio']
    return X, y

def split_data(X, y):
    """Split the data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def find_best_depth(X_train, y_train, X_val, y_val, max_depth_range):
    """Find the best tree depth based on validation accuracy."""
    best_depth = None
    best_accuracy = 0
    train_accuracies = []
    val_accuracies = []

    for depth in max_depth_range:
        model = ID3(max_depth=depth)
        model.tree = model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"Depth: {depth}, Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_depth = depth

    return best_depth, best_accuracy, train_accuracies, val_accuracies

def plot_accuracies(train_accuracies, val_accuracies, max_depth_range):
    """Plot training and validation accuracies, and difference between them to analyze overfitting."""
    # Plot training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(max_depth_range, train_accuracies, label="Training set")
    plt.plot(max_depth_range, val_accuracies, label="Validation set")
    plt.xlabel("Tree depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Comparison of accuracies for different tree depths")
    plt.savefig("results/accuracies.png")
    plt.show()

    # Plot difference to analyze overfitting
    differences = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    plt.figure(figsize=(10, 5))
    plt.plot(max_depth_range, differences, label="Training - Validation")
    plt.xlabel("Tree depth")
    plt.ylabel("Accuracy difference")
    plt.title("Overfitting analysis: Difference between training and validation accuracies")
    plt.legend()
    plt.savefig("results/overfitting_analysis.png")
    plt.show()

def evaluate_on_test_set(X_train, y_train, X_test, y_test, best_depth):
    """Train the final model and evaluate it on the test set."""
    final_model = ID3(max_depth=best_depth)
    final_model.tree = final_model.fit(X_train, y_train)
    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return test_accuracy

def evaluate_bins_effect(file_path, continuous_features, bin_sizes, max_depth):
    """Evaluate the effect of the number of bins (q) on model accuracy."""
    train_accuracies = []
    val_accuracies = []

    for q in bin_sizes:
        X, y = load_and_preprocess_data(file_path, continuous_features, q=q)
        X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

        model = ID3(max_depth=max_depth)
        model.tree = model.fit(X_train.to_numpy(), y_train.to_numpy())

        train_predictions = model.predict(X_train.to_numpy())
        val_predictions = model.predict(X_val.to_numpy())

        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Plot the results
    plt.plot(bin_sizes, train_accuracies, label="Training set")
    plt.plot(bin_sizes, val_accuracies, label="Validation set")
    plt.xlabel("Number of bins (q)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Effect of number of bins on accuracy")
    plt.savefig("results/bins_effect.png")
    plt.show()

def evaluate_training_set_size_effect(X, y, training_sizes, max_depth):
    """Evaluate the effect of training set size on model accuracy."""
    train_accuracies = []
    val_accuracies = []

    for size in training_sizes:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=size, random_state=42)
        X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        model = ID3(max_depth=max_depth)
        model.tree = model.fit(X_train.to_numpy(), y_train.to_numpy())

        train_predictions = model.predict(X_train.to_numpy())
        val_predictions = model.predict(X_val.to_numpy())

        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Plot the results
    plt.plot(training_sizes, train_accuracies, label="Training set")
    plt.plot(training_sizes, val_accuracies, label="Validation set")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Effect of training set size on accuracy")
    plt.savefig("results/training_size_effect.png")
    plt.show()

def main():
    # Configurations
    file_path = 'cardio_train.csv'
    continuous_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    max_depth_range = range(1, 11)

    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path, continuous_features)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Convert to NumPy
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_val_np = X_val.to_numpy()
    y_val_np = y_val.to_numpy()
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    # Find the best depth
    best_depth, best_accuracy, train_accuracies, val_accuracies = find_best_depth(
        X_train_np, y_train_np, X_val_np, y_val_np, max_depth_range
    )
    print(f"Best depth: {best_depth} with validation accuracy: {best_accuracy:.4f}")

    # Plot accuracies
    plot_accuracies(train_accuracies, val_accuracies, max_depth_range)

    # Evaluate on the test set
    test_accuracy = evaluate_on_test_set(X_train_np, y_train_np, X_test_np, y_test_np, best_depth)
    print(f"Test set accuracy: {test_accuracy:.4f}")

    # Evaluate training set size effect
    training_sizes = np.linspace(0.1, 0.9, 9)  # Sizes from 10% to 90% of the data
    evaluate_training_set_size_effect(X, y, training_sizes, max_depth=best_depth)

    # Evaluate bins effect
    bin_sizes = range(2, 21)  # Test different bin sizes (q)
    evaluate_bins_effect(file_path, continuous_features, bin_sizes, max_depth=best_depth)

if __name__ == "__main__":
    main()
