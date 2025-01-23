import numpy as np
from scipy.stats import norm
from solver import Solver

class NaiveBayesClassifier(Solver):
    """
    Naive Bayes Classifier.

    Attributes:
        class_priors (dict): Prior probabilities for each class.
        feature_distributions (dict): Feature distributions for each class.
        continuous_features (list): Indices of features treated as continuous.
    """
    def __init__(self, continuous_features):
        """
        Initialize the Naive Bayes Classifier.

        Args:
            continuous_features (list): List of indices for features to be treated as continuous.
        """
        self.class_priors = {}
        self.feature_distributions = {}
        self.continuous_features = continuous_features

    def fit(self, X, y):
        """
        Fit the Naive Bayes Classifier to the training data.

        Args:
            X (numpy.ndarray): Feature matrix of shape.
            y (numpy.ndarray): Target vector of shape.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(y)
            self.feature_distributions[cls] = {}
            for i in range(X.shape[1]):
                if i in self.continuous_features:
                    self.feature_distributions[cls][i] = (np.mean(X_cls[:, i]), np.std(X_cls[:, i]))
                else:
                    values, counts = np.unique(X_cls[:, i], return_counts=True)
                    self.feature_distributions[cls][i] = dict(zip(values, counts / len(X_cls)))

    def predict(self, X):
        """
        Predict class labels for the given feature matrix.

        Args:
            X (numpy.ndarray): Feature matrix of shape.

        Returns:
            numpy.ndarray: Predicted class labels of shape.
        """
        predictions = []
        for x in X:
            class_probabilities = {}
            for cls in self.classes:
                prior = self.class_priors[cls]
                likelihood = 1
                for i in range(len(x)):
                    if i in self.continuous_features:
                        mean, std = self.feature_distributions[cls][i]
                        likelihood *= norm.pdf(x[i], loc=mean, scale=std)
                    else:
                        likelihood *= self.feature_distributions[cls][i].get(x[i], 1e-6)
                class_probabilities[cls] = prior * likelihood
            predictions.append(max(class_probabilities, key=class_probabilities.get))
        return np.array(predictions)
