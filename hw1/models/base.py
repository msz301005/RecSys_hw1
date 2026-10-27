"""Shared deterministic recommendation policy, independent of model targets."""
import numpy as np


class TopKRecommender:
    def recommend(self, users, seen, k=10, batch_size=128):
        users = np.asarray(users, dtype=np.int64)
        if np.any(users < 0) or np.any(users >= seen.shape[0]):
            raise ValueError("Cold users are excluded by the warm-start evaluation protocol")
        result = []
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            scores = np.asarray(self.score_users(batch), dtype=np.float64)
            if scores.shape != (len(batch), seen.shape[1]) or not np.isfinite(scores).all():
                raise ValueError("Invalid score matrix")
            history = seen[batch].tocoo()
            scores[history.row, history.col] = -np.inf
            if np.any(np.isfinite(scores).sum(axis=1) < k):
                raise ValueError("Fewer than K unseen candidates")
            # Stable full sort: equal scores are resolved by the ascending internal item ID.
            result.append(np.argsort(-scores, axis=1, kind="stable")[:, :k])
        return np.vstack(result) if result else np.empty((0, k), dtype=np.int64)
