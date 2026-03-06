import numpy as np


class GaussianBayesClassifier:

    def __init__(self):
        self.classes = None
        self.pi = None
        self.mu = None
        self.sigma = None

    def fit(self, X, y):

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.pi = np.zeros(n_classes)
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))

        for c in self.classes:

            X_c = X[y == c]

            # class prior
            self.pi[int(c)] = X_c.shape[0] / X.shape[0]

            # class mean
            self.mu[int(c)] = np.mean(X_c, axis=0)

            # covariance matrix
            diff = X_c - self.mu[int(c)]
            self.sigma[int(c)] = np.dot(diff.T, diff) / X_c.shape[0]

    def gaussian_pdf(self, x, mean, cov):

        n = len(mean)

        diff = x - mean
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        numerator = np.exp(-0.5 * np.dot(np.dot(diff.T, inv), diff))
        denominator = np.sqrt(((2 * np.pi) ** n) * det)

        return numerator / denominator

    def predict(self, X):

        predictions = []

        for x in X:

            probs = []

            for c in self.classes:

                prior = self.pi[int(c)]
                likelihood = self.gaussian_pdf(x, self.mu[int(c)], self.sigma[int(c)])

                posterior = prior * likelihood
                probs.append(posterior)

            predictions.append(np.argmax(probs))

        return np.array(predictions)