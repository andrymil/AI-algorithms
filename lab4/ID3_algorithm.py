import numpy as np
from solver import Solver

class ID3(Solver):
    """
    Class for ID3 algorithm.

    Attributes:
        max_depth (int): the maximum depth of the tree
        tree (dict): the decision tree
    """

    def __init__(self, max_depth=None):
        """
        The constructor for ID3 class.

        Parameters:
            max_depth (int): the maximum depth of the tree
        """
        self.max_depth = max_depth
        self.tree = None

    def get_parameters(self):
        """
        Get the hyperparameters.

        Returns:
            dict: the hyperparameters
        """
        return {'max_depth': self.max_depth}

    def entropy(self, y):
        """
        Calculate the entropy of the dataset.

        Parameters:
            y (np.ndarray): the class labels

        Returns:
            float: the entropy
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def information_gain(self, X, y, feature):
        """
        Calculate the information gain for a given attribute.

        Parameters:
            X (np.ndarray): the dataset
            y (np.ndarray): the class labels
            feature (int): the attribute index

        Returns:
            float: the information gain
        """
        total_entropy = self.entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = sum((counts[i] / len(y)) * self.entropy(y[X[:, feature] == v])
                               for i, v in enumerate(values))
        return total_entropy - weighted_entropy

    def fit(self, X, y, depth=0):
        """
        Build the decision tree.

        Parameters:
            X (np.ndarray): the dataset
            y (np.ndarray): the class labels
            depth (int): the current depth of the tree

        Returns:
            dict: the decision tree
        """
        majority_class = np.bincount(y).argmax()

        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return majority_class
        
        best_feature = np.argmax([self.information_gain(X, y, i) for i in range(X.shape[1])])
        tree = {best_feature: {"majority_class": majority_class, "branches": {}}}
        
        for value in range(int(np.max(X[:, best_feature])) + 1):
            subset_X = X[X[:, best_feature] == value]
            subset_y = y[X[:, best_feature] == value]

            if len(subset_y) == 0:                
                tree[best_feature]["branches"][value] = majority_class
            else:                
                tree[best_feature]["branches"][value] = self.fit(subset_X, subset_y, depth + 1)

        return tree

    def predict_one(self, x, tree):
        """
        Predict the class for a single example.

        Parameters:
            x (np.ndarray): the example
            tree (dict): the decision tree

        Returns:
            any: the predicted class
        """
        if not isinstance(tree, dict):
            return tree

        feature = list(tree.keys())[0]
        node = tree[feature]
        value = x[feature]

        if value in node["branches"]:
            return self.predict_one(x, node["branches"][value])
        else:
            return node["majority_class"]

    def predict(self, X):
        """
        Predict the class for multiple examples.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            np.ndarray: the predicted classes
        """
        return np.array([self.predict_one(x, self.tree) for x in X])
