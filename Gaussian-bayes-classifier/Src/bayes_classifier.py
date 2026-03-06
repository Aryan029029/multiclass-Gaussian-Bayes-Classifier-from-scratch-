import numpy as np
import pandas as pd


class GaussianNaiveBayes:

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Train the model by calculating mean, variance and priors
        """
        pass

    def predict(self, X):
        """
        Predict class labels for given data
        """
        pass