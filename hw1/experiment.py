"""Validation-only model selection, frozen settings, and a single final test pass."""
from pathlib import Path
import gc
import importlib.metadata
import json
import platform
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from dataset import matrices
from features import build_features
from evaluation import evaluate, bootstrap_interval
from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine, CustomTargetALS

SEED = 42
PRIMARY_METRIC = "NDCG@10"
GRIDS = {
    "ItemKNN": [{"neighbors": k, "shrinkage": 10.0} for k in [50, 100, 200]],
    "iALS": [{"factors": f, "regularization": r, "alpha": 20., "iterations": 12, "seed": SEED}
             for f in [32, 64] for r in [1., 5.]],
    "EASE": [{"regularization": r} for r in [50., 200., 1000.]],
    "FM+metadata": [{"factors": f, "learning_rate": lr, "epochs": 10, "regularization": 0.0001,
                     "negatives": 3, "batch_size": 2048, "seed": SEED} for f, lr in [(16, .03), (16, .08), (32, .05)]],
}


def fit_model(name, config, frame, dataset, cutoff):
    matrices_ = matrices(frame, dataset.shape)
    if name == "ItemKNN":
        model = ItemKNN(**config).fit(matrices_["positive"])
    elif name == "EASE":
        model = EASE(**config).fit(matrices_["positive"])
    elif name == "iALS":
        model = ImplicitALS(**config).fit(matrices_["observed"], matrices_["target"])
    elif name in ["Custom ALS", "Custom ALS (rating only)"]:
        model = CustomTargetALS(**config).fit(matrices_["observed"], matrices_["ratings"], matrices_["timestamps"], cutoff)
    elif name in ["FM+metadata", "FM (IDs only)"]:
        user_features, item_features, n_features = build_features(matrices_["observed"], dataset.metadata, *dataset.shape,
                                                                 use_metadata=name == "FM+metadata")
        model = FactorizationMachine(**config).fit(matrices_["observed"], matrices_["positive"], user_features, item_features, n_features)
    else:
        raise ValueError(name)
    return model, matrices_["observed"]


def environment():
    packages = ["numpy", "scipy", "pandas", "matplotlib", "nbformat", "nbclient", "threadpoolctl", "ipykernel"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {p: importlib.metadata.version(p) for p in packages},
            "seed": SEED, "blas_threads": 1}


def run_validation(dataset, output="artifacts"):
    output = Path(output)
    output.mkdir(exist_ok=True)
    rows, selected, curves = [], {}, {}

    def search(name, configurations):
        candidates = []
        for index, config in enumerate(configurations):
            print(f"Validation: {name} {index + 1}/{len(configurations)} {config}", flush=True)
            start = time.perf_counter()
            model, seen = fit_model(name, config, dataset.train, dataset, dataset.train_cutoff)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            metrics, details = evaluate(model, dataset.validation, seen)
            evaluation_seconds = time.perf_counter() - start
            row = {"model": name, "config_id": index, "config": config, **metrics,
                   "fit_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds}
            rows.append(row)
            candidates.append(row)
            curves[f"{name}/{index}"] = getattr(model, "loss_history", [])
            print(f"  {PRIMARY_METRIC}={metrics[PRIMARY_METRIC]:.6f}, Recall@10={metrics['Recall@10']:.6f}, fit={fit_seconds:.1f}s", flush=True)
            del model, details
            gc.collect()
        # Deterministic tie-break: the earlier declared configuration wins.
        best = max(candidates, key=lambda r: r[PRIMARY_METRIC])
        selected[name] = best["config"]

    with threadpool_limits(limits=1):
        for name, configurations in GRIDS.items():
            search(name, configurations)
        # Baseline-selected ALS settings are reused: confidence and capacity stay the same.
        custom_grid = [{**selected["iALS"], "half_life_days": h, "freshness_floor": .5} for h in [180., 720.]]
        search("Custom ALS", custom_grid)
        # Controls are fixed from the corresponding main model before any test evaluation.
        search("FM (IDs only)", [selected["FM+metadata"].copy()])
        search("Custom ALS (rating only)", [{**selected["iALS"], "half_life_days": float(1e12), "freshness_floor": 1.0}])
    pd.DataFrame(rows).to_csv(output / "validation_search.csv", index=False)
    frozen = {"primary_metric": PRIMARY_METRIC, "selected": selected, "grids": {**GRIDS, "Custom ALS": custom_grid},
              "protocol": dataset.diagnostics, "environment": environment(), "curves": curves,
              "selection_uses_test": False}
    # This artifact is written before run_test is called.
    (output / "selection.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    return pd.DataFrame(rows), frozen


def run_test(dataset, frozen, output="artifacts"):
    if frozen["protocol"]["split_sha256"] != dataset.diagnostics["split_sha256"]:
        raise ValueError("Frozen selection belongs to another dataset or split")
    output = Path(output)
    train_val = pd.concat([dataset.train, dataset.validation], ignore_index=True)
    rows, details, curves, example_rows = [], {}, {}, []
    reference_users = None
    with threadpool_limits(limits=1):
        for name, config in frozen["selected"].items():
            print(f"Final train+validation fit: {name}", flush=True)
            start = time.perf_counter()
            model, seen = fit_model(name, config, train_val, dataset, dataset.test_cutoff)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            metrics, per_user = evaluate(model, dataset.test, seen)
            evaluation_seconds = time.perf_counter() - start
            if reference_users is None:
                reference_users = per_user["users"]
            np.testing.assert_array_equal(reference_users, per_user["users"])
            confidence_interval = bootstrap_interval(per_user["per_user"][:, 1])
            row = {"model": name, **metrics, "NDCG_CI_low": confidence_interval[0], "NDCG_CI_high": confidence_interval[1],
                   "fit_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds, "config": config}
            rows.append(row)
            details[name] = per_user
            curves[name] = getattr(model, "loss_history", [])
            for rank, item in enumerate(per_user["recommendations"][0], start=1):
                example_rows.append({"model": name, "anonymous_user": int(reference_users[0]), "rank": rank,
                                     "title": dataset.metadata[item]["title"], "brand": dataset.metadata[item]["brand"]})
            print(f"  test NDCG@10={metrics['NDCG@10']:.6f}, Recall@10={metrics['Recall@10']:.6f}", flush=True)
            del model
            gc.collect()
    table = pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False)
    table.to_csv(output / "test_results.csv", index=False)
    table.to_json(output / "test_results.json", orient="records", indent=2, force_ascii=False)
    pd.DataFrame(example_rows).to_csv(output / "recommendation_examples.csv", index=False)
    paired = {}
    for a, b in [("Custom ALS", "iALS"), ("FM+metadata", "FM (IDs only)"), ("Custom ALS", "Custom ALS (rating only)")]:
        difference = details[a]["per_user"][:, 1] - details[b]["per_user"][:, 1]
        paired[f"{a} minus {b}"] = {"mean_ndcg_difference": float(difference.mean()), "paired_bootstrap_95pct": bootstrap_interval(difference)}
    report = {"test_results": table.to_dict(orient="records"), "paired_comparisons": paired,
              "final_loss_curves": curves, "test_users": len(reference_users),
              "test_relevance": "rating >= 4", "candidate_policy": "fixed train-defined catalogue minus all prior observed items",
              "test_used_for_selection": False, "source_manifest": json.loads(Path("data/download_manifest.json").read_text())}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return table, report, pd.DataFrame(example_rows)
