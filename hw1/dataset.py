"""Streaming input, deterministic train-only cohort selection and global time split."""
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Dataset:
    raw: pd.DataFrame
    clean: pd.DataFrame
    selected: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: list
    user_ids: list
    item_ids: list
    train_cutoff: int
    test_cutoff: int
    diagnostics: dict

    @property
    def shape(self):
        return len(self.user_ids), len(self.item_ids)


def matrices(frame, shape):
    coords = (frame.user.to_numpy(), frame.item.to_numpy())
    obs = sparse.csr_matrix((np.ones(len(frame)), coords), shape=shape)
    ratings = sparse.csr_matrix((frame.rating.to_numpy(dtype=float), coords), shape=shape)
    times = sparse.csr_matrix((frame.timestamp.to_numpy(dtype=float), coords), shape=shape)
    positive = sparse.csr_matrix(((frame.rating.to_numpy() >= 4).astype(float), coords), shape=shape)
    positive.eliminate_zeros()
    target = obs.copy()
    target.data = (ratings.data >= 4).astype(float)
    return {"observed": obs, "ratings": ratings, "timestamps": times, "positive": positive, "target": target}


def prepare_data(directory="data", max_items=3000, min_train_history=3):
    directory = Path(directory)
    rows = []
    with gzip.open(directory / "Video_Games_5.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append((row.get("reviewerID"), row.get("asin"), row.get("overall"), row.get("unixReviewTime")))
    raw = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    missing = raw.isna().sum().astype(int).to_dict()
    valid = raw.dropna().copy()
    valid = valid[valid.rating.between(1, 5) & (valid.timestamp > 0)]
    valid.timestamp = valid.timestamp.astype(np.int64)
    duplicate_pairs = int(valid.duplicated(["user_id", "item_id"]).sum())
    # Earliest event only: never replace a historical rating using a future re-review.
    clean = valid.sort_values("timestamp", kind="stable").drop_duplicates(["user_id", "item_id"], keep="first").copy()
    train_cutoff, test_cutoff = [int(x // 86400 * 86400) for x in np.quantile(clean.timestamp, [0.7, 0.85])]
    before = clean[clean.timestamp < train_cutoff]
    counts = before.groupby("item_id").size().rename("n").reset_index()
    catalog = counts.sort_values(["n", "item_id"], ascending=[False, True]).head(max_items).item_id.tolist()
    before = before[before.item_id.isin(catalog)]
    activity = before.groupby("user_id").size()
    users = sorted(activity[activity >= min_train_history].index)
    # Remove catalogue items with zero training support after the user filter.
    catalog = sorted(before[before.user_id.isin(users)].item_id.unique())
    selected = clean[clean.user_id.isin(users) & clean.item_id.isin(catalog)].copy()
    user_index, item_index = {u: i for i, u in enumerate(users)}, {i: j for j, i in enumerate(catalog)}
    selected["user"] = selected.user_id.map(user_index).astype(np.int64)
    selected["item"] = selected.item_id.map(item_index).astype(np.int64)
    selected = selected.sort_values(["timestamp", "user", "item"], kind="stable").reset_index(drop=True)
    train = selected[selected.timestamp < train_cutoff].copy()
    validation = selected[(selected.timestamp >= train_cutoff) & (selected.timestamp < test_cutoff)].copy()
    test = selected[selected.timestamp >= test_cutoff].copy()
    wanted, meta = set(catalog), {}
    meta_rows = meta_duplicates = 0
    with gzip.open(directory / "meta_Video_Games.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta_rows += 1
            asin = row.get("asin")
            if asin in wanted:
                if asin in meta:
                    meta_duplicates += 1
                else:
                    meta[asin] = {key: row.get(key, "" if key != "category" else []) for key in ["title", "brand", "category"]}
    metadata = [meta.get(asin, {"title": asin, "brand": "", "category": []}) for asin in catalog]
    split_stats = {}
    for name, lower, upper in [("validation", train_cutoff, test_cutoff), ("test", test_cutoff, np.inf)]:
        phase = clean[(clean.timestamp >= lower) & (clean.timestamp < upper)]
        split_stats[name] = {"original_events": len(phase), "cold_user_events": int((~phase.user_id.isin(users)).sum()),
                             "outside_catalog_events": int((~phase.item_id.isin(catalog)).sum()),
                             "retained_events": len(validation if name == "validation" else test)}
    split_fingerprint = hashlib.sha256(selected[["user", "item", "rating", "timestamp"]].to_csv(index=False).encode()).hexdigest()
    diagnostics = {"raw_reviews": len(raw), "raw_users": raw.user_id.nunique(), "raw_items": raw.item_id.nunique(),
                   "missing": missing, "invalid_rows": len(raw) - len(valid), "duplicate_pairs_removed": duplicate_pairs,
                   "clean_reviews": len(clean), "selected_reviews": len(selected), "users": len(users), "items": len(catalog),
                   "train_events": len(train), "validation_events": len(validation), "test_events": len(test),
                   "train_cutoff_utc": pd.to_datetime(train_cutoff, unit="s", utc=True).isoformat(),
                   "test_cutoff_utc": pd.to_datetime(test_cutoff, unit="s", utc=True).isoformat(),
                   "sparsity": 1 - len(selected) / (len(users) * len(catalog)), "metadata_raw_rows": meta_rows,
                   "metadata_duplicate_selected_items": meta_duplicates, "metadata_items_found": len(meta),
                   "metadata_missing_brand": sum(not r["brand"] for r in metadata),
                   "metadata_missing_category": sum(not r["category"] for r in metadata),
                   "metadata_missing_title": sum(not r["title"] for r in metadata),
                   "phase_coverage": split_stats, "split_sha256": split_fingerprint,
                   "max_items": max_items, "min_train_history": min_train_history}
    if not (100_000 <= len(selected) <= 2_000_000 and len(users) >= 1000 and len(catalog) >= 1000):
        raise ValueError(f"Dataset does not meet assignment size constraints: {diagnostics}")
    assert train.timestamp.max() < train_cutoff <= validation.timestamp.min()
    assert validation.timestamp.max() < test_cutoff <= test.timestamp.min()
    assert not selected.duplicated(["user", "item"]).any()
    return Dataset(raw, clean, selected, train, validation, test, metadata, users, catalog,
                   train_cutoff, test_cutoff, diagnostics)
