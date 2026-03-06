import numpy as np
from Src.bayes_classifier import GaussianBayesClassifier

# Load data
X_train = np.genfromtxt("Data/X_train-3.csv", delimiter=",")
y_train = np.genfromtxt("Data/y_train-2.csv")
X_test = np.genfromtxt("Data/X_test_all-1.csv", delimiter=",")

# Create classifier
classifier = GaussianBayesClassifier()

# Train model
classifier.fit(X_train, y_train)

# Predict test data
y_pred = classifier.predict(X_test)

# Print predictions
print(y_pred)

# Save predictions
np.savetxt("results/predictions.csv", y_pred, delimiter=",", fmt="%d")

print("Predictions saved to results/predictions.csv")