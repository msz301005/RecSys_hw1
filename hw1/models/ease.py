"""EASE closed-form ridge regression with a zero diagonal."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import TopKRecommender


class EASE(TopKRecommender):
    def __init__(self, regularization=200.0):
        self.regularization = regularization

    def fit(self, positive):
        if self.regularization <= 0:
            raise ValueError("Regularization must be positive")
        self.history = positive.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.regularization
        inverse = cho_solve(cho_factor(gram, lower=True), np.eye(gram.shape[0]))
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score_users(self, users):
        return np.asarray(self.history[users] @ self.weights)
