from ID3_algorithm import ID3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def load_and_preprocess_data(file_path, continuous_features):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path, sep=';')
    for feature in continuous_features:
        data[feature] = pd.qcut(data[feature], q=10, labels=False, duplicates='drop')
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
        print(f"Głębokość: {depth}, Dokładność treningowa: {train_accuracy:.4f}, Dokładność walidacyjna: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_depth = depth

    return best_depth, best_accuracy, train_accuracies, val_accuracies

def plot_accuracies(train_accuracies, val_accuracies, max_depth_range, output_file):
    """Plot training and validation accuracies."""
    plt.plot(max_depth_range, train_accuracies, label="Zbiór treningowy")
    plt.plot(max_depth_range, val_accuracies, label="Zbiór walidacyjny")
    plt.xlabel("Głębokość drzewa")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.title("Porównanie dokładności dla różnych głębokości drzewa")
    plt.savefig(output_file)
    plt.show()

def evaluate_on_test_set(X_train, y_train, X_test, y_test, best_depth):
    """Train the final model and evaluate it on the test set."""
    final_model = ID3(max_depth=best_depth)
    final_model.tree = final_model.fit(X_train, y_train)
    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return test_accuracy

def main():
    # Configurations
    file_path = 'cardio_train.csv'
    continuous_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    max_depth_range = range(1, 11)
    output_file = "accuracy_plot.png"

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
    print(f"Najlepsza głębokość: {best_depth} z dokładnością walidacyjną: {best_accuracy:.4f}")

    # Plot accuracies
    plot_accuracies(train_accuracies, val_accuracies, max_depth_range, output_file)

    # Evaluate on the test set
    test_accuracy = evaluate_on_test_set(X_train_np, y_train_np, X_test_np, y_test_np, best_depth)
    print(f"Dokładność na zbiorze testowym: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
