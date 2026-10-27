"""Second-order FM, manual gradients, AdaGrad, and sampled logistic training."""
import numpy as np
from scipy.special import expit
from .base import TopKRecommender


class FactorizationMachine(TopKRecommender):
    def __init__(self, factors=16, learning_rate=0.05, regularization=0.0001,
                 epochs=10, negatives=3, batch_size=2048, seed=42):
        self.factors, self.learning_rate, self.regularization = factors, learning_rate, regularization
        self.epochs, self.negatives, self.batch_size, self.seed = epochs, negatives, batch_size, seed

    def predict_fields(self, fields):
        vectors = self.v[fields]
        summed = vectors.sum(axis=1)
        return self.bias + self.w[fields].sum(axis=1) + 0.5 * (summed ** 2 - (vectors ** 2).sum(axis=1)).sum(axis=1)

    def batch_gradients(self, fields, labels):
        """BCE and gradients, including L2 on distinct active parameters."""
        vectors = self.v[fields]
        summed = vectors.sum(axis=1)
        score = self.predict_fields(fields)
        error = (expit(score) - labels) / len(labels)
        unique, inverse = np.unique(fields, return_inverse=True)
        inverse = inverse.reshape(fields.shape)
        grad_w = np.zeros(len(unique))
        grad_v = np.zeros((len(unique), self.factors))
        np.add.at(grad_w, inverse.ravel(), np.broadcast_to(error[:, None], fields.shape).ravel())
        individual = error[:, None, None] * (summed[:, None, :] - vectors)
        np.add.at(grad_v, inverse.ravel(), individual.reshape(-1, self.factors))
        grad_w += self.regularization * self.w[unique]
        grad_v += self.regularization * self.v[unique]
        loss = np.mean(np.logaddexp(0, score) - labels * score)
        loss += 0.5 * self.regularization * (np.square(self.w[unique]).sum() + np.square(self.v[unique]).sum())
        return float(loss), float(error.sum()), unique, grad_w, grad_v

    def fit(self, observed, positive, user_features, item_features, n_features):
        self.user_features, self.item_features = user_features, item_features
        rng = np.random.default_rng(self.seed)
        self.bias = 0.0
        self.w = np.zeros(n_features)
        self.v = rng.normal(0, 0.03, (n_features, self.factors))
        acc_w, acc_v, acc_b = np.full_like(self.w, 1e-8), np.full_like(self.v, 1e-8), 1e-8
        obs = observed.tocoo()
        pos = positive.tocoo()
        observed_codes = obs.row.astype(np.int64) * observed.shape[1] + obs.col
        positive_codes = pos.row.astype(np.int64) * observed.shape[1] + pos.col
        original_labels = np.isin(observed_codes, positive_codes).astype(float)
        self.loss_history = []
        for _ in range(self.epochs):
            negative_users = np.repeat(pos.row, self.negatives)
            negative_items = rng.integers(0, observed.shape[1], len(negative_users))
            # Rejection uses only observed TRAIN interactions, never withheld labels.
            bad = np.isin(negative_users.astype(np.int64) * observed.shape[1] + negative_items, observed_codes)
            while bad.any():
                negative_items[bad] = rng.integers(0, observed.shape[1], bad.sum())
                bad = np.isin(negative_users.astype(np.int64) * observed.shape[1] + negative_items, observed_codes)
            train_u = np.concatenate([obs.row, negative_users])
            train_i = np.concatenate([obs.col, negative_items])
            labels = np.concatenate([original_labels, np.zeros(len(negative_users))])
            order = rng.permutation(len(labels))
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                fields = np.column_stack([user_features[train_u[ids]], item_features[train_i[ids]]])
                loss, grad_b, unique, grad_w, grad_v = self.batch_gradients(fields, labels[ids])
                acc_b += grad_b ** 2
                acc_w[unique] += grad_w ** 2
                acc_v[unique] += grad_v ** 2
                self.bias -= self.learning_rate * grad_b / np.sqrt(acc_b)
                self.w[unique] -= self.learning_rate * grad_w / np.sqrt(acc_w[unique])
                self.v[unique] -= self.learning_rate * grad_v / np.sqrt(acc_v[unique])
                total_loss += loss * len(ids)
            self.loss_history.append(total_loss / len(labels))
        self._cache_scores()
        return self

    def _field_terms(self, fields):
        vectors = self.v[fields]
        summed = vectors.sum(axis=1)
        offset = self.w[fields].sum(axis=1) + 0.5 * (summed ** 2 - (vectors ** 2).sum(axis=1)).sum(axis=1)
        return summed, offset

    def _cache_scores(self):
        self.user_vectors, self.user_offsets = self._field_terms(self.user_features)
        self.item_vectors, self.item_offsets = self._field_terms(self.item_features)

    def score_users(self, users):
        # Algebraically identical to evaluating every full FM feature vector.
        return self.bias + self.user_offsets[users, None] + self.item_offsets[None, :] + self.user_vectors[users] @ self.item_vectors.T
