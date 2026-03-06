import numpy as np
import sys
import os

sys.path.append("src")

from bayes_classifier import GaussianBayesClassifier

X_train = np.genfromtxt("Data/X_train-3.csv", delimiter=",")
y_train = np.genfromtxt("Data/y_train-2.csv", delimiter=",")
X_test = np.genfromtxt("Data/X_test_all-1.csv", delimiter=",")

model = GaussianBayesClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)