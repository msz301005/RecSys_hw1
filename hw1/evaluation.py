"""Full-catalog macro metrics and user-level uncertainty, without sampled candidates."""
import numpy as np


def ranking_metrics(recommendations, truth):
    """Return per-user Recall, binary NDCG, and Precision at recommendation width K."""
    k = recommendations.shape[1]
    discount = 1.0 / np.log2(np.arange(k) + 2)
    rows = []
    for rec, relevant in zip(recommendations, truth, strict=True):
        relevant = set(relevant)
        if not relevant:
            raise ValueError("Macro metrics require at least one relevant held-out item")
        if len(set(rec)) != len(rec):
            raise ValueError("Duplicate recommendations")
        hit = np.asarray([int(item in relevant) for item in rec])
        ideal = discount[:min(k, len(relevant))].sum()
        rows.append([hit.sum() / len(relevant), float(hit @ discount / ideal), hit.mean()])
    return np.asarray(rows, dtype=float)


def evaluate(model, heldout, seen, k=10):
    positive = heldout[heldout.rating >= 4]
    grouped = positive.groupby("user", sort=True).item.apply(list)
    users = grouped.index.to_numpy(dtype=int)
    if not len(users):
        raise ValueError("No evaluable users")
    recommendations = model.recommend(users, seen, k=k)
    for user, rec in zip(users, recommendations, strict=True):
        if seen[user, rec].nnz:
            raise AssertionError("Known item leaked into recommendations")
    per_user = ranking_metrics(recommendations, grouped.tolist())
    scores = dict(zip(["Recall@10", "NDCG@10", "Precision@10"], per_user.mean(axis=0), strict=True))
    scores.update({"users": len(users), "Coverage@10": len(np.unique(recommendations)) / seen.shape[1]})
    return scores, {"users": users, "recommendations": recommendations, "per_user": per_user}


def bootstrap_interval(values, seed=42, repetitions=1000):
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(repetitions)])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]
