import numpy as np


class GaussianBayesClassifier:

    def __init__(self):
        self.classes = None
        self.pi = None        # class priors
        self.mu = None        # class means
        self.sigma = None     # class covariance matrices

    def fit(self, X, y):
        """
        Train the classifier by computing:
        - class prior probabilities
        - class means
        - class covariance matrices
        """

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.pi = np.zeros(n_classes)
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))

        for c in self.classes:

            # Get all samples belonging to class c
            X_c = X[y == c]

            # Prior probability P(Y=c)
            self.pi[int(c)] = X_c.shape[0] / X.shape[0]

            # Mean vector
            self.mu[int(c)] = np.mean(X_c, axis=0)

            # Covariance matrix
            diff = X_c - self.mu[int(c)]
            self.sigma[int(c)] = np.dot(diff.T, diff) / X_c.shape[0]