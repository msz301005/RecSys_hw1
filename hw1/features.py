"""Fixed sparse categorical fields; no validation/test interaction statistics."""
import hashlib
import html
import re
import numpy as np


def stable_bucket(value, buckets):
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def build_features(history, metadata, n_users, n_items, use_metadata=True):
    """Metadata belongs only to the train-defined catalogue, ID order is fixed."""
    users = np.arange(n_users, dtype=np.int64)[:, None]
    items = (n_users + np.arange(n_items, dtype=np.int64))[:, None]
    offset = n_users + n_items
    if not use_metadata:
        return users, items, offset
    activity = np.minimum(7, np.floor(np.log2(1 + np.asarray(history.sum(axis=1)).ravel())).astype(int))
    users = np.column_stack([users, offset + activity])
    offset += 8
    specifications = [("brand", 512), ("platform", 128), ("word1", 512), ("word2", 512)]
    item_columns = [items[:, 0]]
    for field, buckets in specifications:
        values = []
        for row in metadata:
            categories = row.get("category", [])
            if categories and isinstance(categories[0], list):
                categories = categories[0]
            tokens = re.findall(r"[a-z]{3,}", html.unescape(row.get("title", "")).lower())
            value = {
                "brand": row.get("brand") or "<missing>",
                "platform": categories[1] if len(categories) > 1 else "<missing>",
                "word1": tokens[0] if tokens else "<missing>",
                "word2": tokens[1] if len(tokens) > 1 else "<missing>",
            }[field]
            values.append(offset + stable_bucket(value, buckets))
        item_columns.append(values)
        offset += buckets
    return users, np.column_stack(item_columns).astype(np.int64), offset
