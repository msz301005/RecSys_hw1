"""Item-based cosine nearest neighbours with significance shrinkage."""
import numpy as np
from scipy import sparse
from .base import TopKRecommender


class ItemKNN(TopKRecommender):
    def __init__(self, neighbors=100, shrinkage=10.0):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def fit(self, positive):
        self.history = positive.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(gram))
        denom = norms[:, None] * norms[None, :] + self.shrinkage
        sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        np.fill_diagonal(sim, 0.0)
        # Each target column retains its K most similar source items.
        count = min(self.neighbors, max(0, sim.shape[0] - 1))
        rows, cols, values = [], [], []
        for j in range(sim.shape[1]):
            ids = np.argsort(-sim[:, j], kind="stable")[:count]
            ids = ids[sim[ids, j] > 0]
            rows.extend(ids)
            cols.extend([j] * len(ids))
            values.extend(sim[ids, j])
        self.weights = sparse.csr_matrix((values, (rows, cols)), shape=sim.shape)
        return self

    def score_users(self, users):
        return (self.history[users] @ self.weights).toarray()
