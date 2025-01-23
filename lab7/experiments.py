import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from bayes import NaiveBayesClassifier

def load_and_preprocess_data(file_path, continuous_features, q=10):
    """Load and preprocess the data with a specific number of bins (q)."""
    data = pd.read_csv(file_path, sep=';')
    for feature in continuous_features:
        data.iloc[:, feature] = pd.qcut(data.iloc[:, feature], q=q, labels=False, duplicates='drop')
    X = data.drop(columns=['id', 'cardio'])
    y = data['cardio']
    return X.values, y.values

def evaluate_model(X, y, classifier, test_size=0.2):
    """Evaluate the model using train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    return train_accuracy, test_accuracy

def cross_validate_model(X, y, classifier, k=5):
    """Evaluate the model using k-fold cross-validation."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
    return np.mean(accuracies), np.std(accuracies)

def train_test_evaluation(X, y, classifier, splits):
    """Perform evaluation for train/test split with multiple splits."""
    train_accuracies = []
    test_accuracies = []
    test_std_devs = []

    for split in splits:
        train_accs = []
        test_accs = []
        for _ in range(10):
            train_acc, test_acc = evaluate_model(X, y, classifier, test_size=split)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        train_accuracies.append(np.mean(train_accs))
        test_accuracies.append(np.mean(test_accs))
        test_std_devs.append(np.std(test_accs))

    return train_accuracies, test_accuracies, test_std_devs

def cross_validation_evaluation(X, y, classifier, k_values):
    """Perform k-fold cross-validation evaluation with multiple fold values."""
    cv_accuracies = []
    cv_stds = []

    for k in k_values:
        mean_acc, std_acc = cross_validate_model(X, y, classifier, k=k)
        cv_accuracies.append(mean_acc)
        cv_stds.append(std_acc)

    return cv_accuracies, cv_stds

def compare_methods(test_accuracies, test_std_devs, splits, cv_accuracies, cv_stds, k_values):
    """Compare train/test split and cross-validation methods."""
    best_split_idx = np.argmax(test_accuracies)
    best_split = splits[best_split_idx]
    best_split_acc = test_accuracies[best_split_idx]
    best_split_std = test_std_devs[best_split_idx]

    best_cv_idx = np.argmax(cv_accuracies)
    best_k = k_values[best_cv_idx]
    best_cv_acc = cv_accuracies[best_cv_idx]
    best_cv_std = cv_stds[best_cv_idx]

    print("Comparison of Methods:")
    print(f"Train/Test Split: Test Size = {best_split}, Accuracy = {best_split_acc:.4f} ± {best_split_std:.4f}")
    print(f"Cross-Validation: Folds = {best_k}, Accuracy = {best_cv_acc:.4f} ± {best_cv_std:.4f}")

    methods = ['Train/Test Split', 'Cross-Validation']
    accuracies = [best_split_acc, best_cv_acc]
    stds = [best_split_std, best_cv_std]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, accuracies, yerr=stds, capsize=10, color=['orange', 'blue'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Comparison of Best Results')
    plt.savefig('comparison_of_methods.png')
    plt.close()

def final_test_evaluation(X, y, classifier):
    """Evaluate the classifier on the final test set."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    classifier.fit(X_train, y_train)
    final_test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f"Final Test Set Accuracy: {final_test_accuracy:.4f}")

def main():
    file_path = 'cardio_train.csv'
    continuous_features = [1, 3, 4, 5, 6]
    X, y = load_and_preprocess_data(file_path, continuous_features=continuous_features, q=10)

    nb_classifier = NaiveBayesClassifier(continuous_features=continuous_features)

    splits = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    train_accuracies, test_accuracies, test_std_devs = train_test_evaluation(X, y, nb_classifier, splits)

    plt.figure(figsize=(10, 6))
    plt.errorbar(splits, test_accuracies, yerr=test_std_devs, label='Test Accuracy', fmt='-o', capsize=5)
    plt.plot(splits, train_accuracies, label='Train Accuracy', marker='o')
    plt.title('Train/Test Split Evaluation')
    plt.xlabel('Test Size Fraction')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('train_test_split_evaluation.png')
    plt.close()

    k_values = [2, 3, 5, 7, 10, 15, 20]
    cv_accuracies, cv_stds = cross_validation_evaluation(X, y, nb_classifier, k_values)

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, cv_accuracies, yerr=cv_stds, label='Cross-Validation Accuracy', fmt='-o', capsize=5)
    plt.title('Cross-Validation Evaluation')
    plt.xlabel('Number of Folds (k)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('cross_validation_evaluation.png')
    plt.close()

    compare_methods(test_accuracies, test_std_devs, splits, cv_accuracies, cv_stds, k_values)

    final_test_evaluation(X, y, nb_classifier)

if __name__ == '__main__':
    main()
