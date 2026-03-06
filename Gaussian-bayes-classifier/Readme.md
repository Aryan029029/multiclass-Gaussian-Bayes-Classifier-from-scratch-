# Gaussian Bayes Classifier (From Scratch)

This project implements a **Gaussian Bayes Classifier from scratch in Python** without using machine learning libraries like scikit-learn.

The classifier learns the probability distribution of each class and predicts the most likely class for unseen data.

## Project Structure

Gaussian-bayes-classifier
│
├── Data
│ ├── X_train-3.csv
│ ├── y_train-2.csv
│ └── X_test_all-1.csv
│
├── results
│ └── predictions.csv
│
├── Src
│ └── bayes_classifier.py
│
└── run.py


## How It Works

The model follows the **Gaussian Naive Bayes assumption**.

For each class:

1. Estimate class prior probability  

P(y = k)

2. Estimate class mean vector  

μₖ

3. Estimate covariance matrix  

Σₖ

The probability of a test point belonging to class k is computed using the **multivariate Gaussian distribution**:

P(x|y=k)

The predicted class is the one with the **maximum posterior probability**.

## Running the Project

Run the classifier with:
run.py


This will generate predictions and save them to:
results/predictions.csv


## Technologies Used

- Python
- NumPy
- Probability & Statistics
- Machine Learning Fundamentals

## Goal

The goal of this project is to demonstrate a **full implementation of a probabilistic classifier from scratch**, without relying on machine learning libraries.
