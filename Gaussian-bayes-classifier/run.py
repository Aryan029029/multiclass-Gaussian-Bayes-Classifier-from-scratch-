import numpy as np
from src.bayes_classifier import GaussianBayesClassifier


# Load training data
X_train = np.genfromtxt("data/X_train-3.csv", delimiter=",")
y_train = np.genfromtxt("data/y_train-2.csv")

# Load test data
X_test = np.genfromtxt("data/X_test_all-1.csv", delimiter=",")


# Train classifier
model = GaussianBayesClassifier()
model.fit(X_train, y_train)


# Predict test labels
predictions = model.predict(X_test)


# Save predictions
np.savetxt("results/predictions.csv", predictions, fmt="%d", delimiter=",")

print("Predictions saved to results/predictions.csv")