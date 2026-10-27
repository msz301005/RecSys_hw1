"""Exact alternating least squares with implicit missing entries and custom P."""
import numpy as np
from .base import TopKRecommender


class ImplicitALS(TopKRecommender):
    def __init__(self, factors=32, regularization=5.0, alpha=20.0, iterations=12, seed=42):
        self.factors, self.regularization = factors, regularization
        self.alpha, self.iterations, self.seed = alpha, iterations, seed

    @staticmethod
    def _update(observed, target, fixed, regularization, alpha):
        gram = fixed.T @ fixed
        prior = gram + regularization * np.eye(fixed.shape[1])
        result = np.zeros((observed.shape[0], fixed.shape[1]), dtype=np.float64)
        for row in range(observed.shape[0]):
            start, stop = observed.indptr[row:row + 2]
            ids = observed.indices[start:stop]
            if len(ids):
                vectors = fixed[ids]
                normal = prior + alpha * (vectors.T @ vectors)
                rhs = (1.0 + alpha) * (vectors.T @ target.data[start:stop])
                result[row] = np.linalg.solve(normal, rhs)
        return result

    def fit(self, observed, target=None):
        observed = observed.astype(np.float64).tocsr()
        observed.sort_indices()
        if np.any(observed.data != 1):
            raise ValueError("Observed matrix must be binary")
        target = observed.copy() if target is None else target.astype(np.float64).tocsr()
        target.sort_indices()
        if not (np.array_equal(observed.indptr, target.indptr) and np.array_equal(observed.indices, target.indices)):
            raise ValueError("Target must preserve all observed coordinates, including explicit zeros")
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.05, (observed.shape[1], self.factors))
        transposed, target_t = observed.T.tocsr(), target.T.tocsr()
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = self._update(observed, target, self.item_factors, self.regularization, self.alpha)
            self.item_factors = self._update(transposed, target_t, self.user_factors, self.regularization, self.alpha)
            self.loss_history.append(self.objective(observed, target))
        return self

    def objective(self, observed, target):
        # Sum of squared predictions for ALL pairs, computed without a dense U x I matrix.
        all_squared = np.sum((self.user_factors.T @ self.user_factors) * (self.item_factors.T @ self.item_factors))
        rows = np.repeat(np.arange(observed.shape[0]), np.diff(observed.indptr))
        values = np.einsum("ij,ij->i", self.user_factors[rows], self.item_factors[observed.indices])
        correction = ((1 + self.alpha) * (target.data - values) ** 2 - values ** 2).sum()
        penalty = self.regularization * (np.square(self.user_factors).sum() + np.square(self.item_factors).sum())
        return float(all_squared + correction + penalty)

    def score_users(self, users):
        return self.user_factors[users] @ self.item_factors.T
