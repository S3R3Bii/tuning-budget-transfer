"""Cold-fit random search and successive halving with a separate full-pool audit."""

from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [110, 211, 312, 413, 514, 615]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode("utf-8")
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def parameters():
    values = []
    for rate, leaves, l2, minimum in itertools.product(
        [0.03, 0.1, 0.3], [7, 15, 31], [0, 1], [10, 30]
    ):
        key = f"lr{rate:g}_leaf{leaves}_l2{l2}_min{minimum}"
        values.append(
            {
                "id": key,
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "l2_regularization": l2,
                "min_samples_leaf": minimum,
            }
        )
    return sorted(values, key=lambda c: c["id"])


def candidate_pool(dataset, seed):
    configs = parameters()
    order = np.random.default_rng(seed + dataset).permutation(len(configs))[:18]
    return [configs[int(i)] for i in order]


def group_partition(group, seed):
    bucket = int(hashlib.sha256(f"{group}:{seed}".encode()).hexdigest()[:8], 16) % 10
    return "fit" if bucket < 6 else "validation" if bucket < 8 else "test"


def load_panel(record):
    path = ROOT / record["panel_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["panel_sha256"]
    frame = pd.read_csv(path, float_precision="round_trip")
    assert (
        len(frame) == record["quality"]["selected_rows"]
        and frame["source_row"].is_unique
    )
    return frame


def scale_target(y):
    center = math.fsum(float(v) for v in y) / len(y)
    variance = math.fsum((float(v) - center) ** 2 for v in y) / len(y)
    assert variance > 0 and math.isfinite(variance)
    return center, math.sqrt(variance)


def losses(actual, prediction, variance):
    errors = [float(p) - float(y) for y, p in zip(actual, prediction)]
    mse = math.fsum(e * e for e in errors) / len(errors)
    return {
        "n": len(errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": math.fsum(abs(e) for e in errors) / len(errors),
        "mean_error": math.fsum(errors) / len(errors),
        "normalized_mse": mse / variance,
    }


def fit_predict(config, iterations, x, y, target, seed):
    center, scale = scale_target(y)
    settings = {k: v for k, v in config.items() if k != "id"}
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=iterations,
        max_bins=64,
        max_depth=None,
        early_stopping=False,
        random_state=seed,
        **settings,
    )
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(x, (y - center) / scale)
        pred = model.predict(target) * scale + center
    elapsed = time.perf_counter() - start
    assert np.isfinite(pred).all()
    return pred, {
        "target_center": center,
        "target_scale": scale,
        "actual_iterations": int(model.n_iter_),
        "elapsed_seconds": elapsed,
    }


def promote(trials, count):
    return sorted(trials, key=lambda t: (t["mse"], t["candidate"]))[:count]


def run_search(record, seed):
    frame = load_panel(record)
    part = frame["group_id"].map(lambda x: group_partition(x, seed))
    target = record["target"]
    features = record["features"]
    for split in ["fit", "validation", "test"]:
        assert int((part == split).sum()) >= 200
    fit = frame.loc[part == "fit"]
    val = frame.loc[part == "validation"]
    test = frame.loc[part == "test"]
    assert (
        set(fit.group_id).isdisjoint(val.group_id)
        and set(fit.group_id).isdisjoint(test.group_id)
        and set(val.group_id).isdisjoint(test.group_id)
    )
    folder = ROOT / f'results/runs/{record["id"]}/{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "selection.json").exists():
        raise RuntimeError(
            "Selection already exists; preserve it instead of overwriting a run."
        )
    split_rows = [
        {
            "source_row": int(row.source_row),
            "group_id": row.group_id,
            "split": str(part.loc[i]),
        }
        for i, row in frame.iterrows()
    ]
    write_csv(folder / "splits.csv", split_rows)
    x = fit[features].to_numpy(dtype=float)
    y = fit[target].to_numpy(dtype=float)
    vx = val[features].to_numpy(dtype=float)
    vy = val[target].to_numpy(dtype=float)
    _, scale = scale_target(y)
    pool = candidate_pool(record["id"], seed)
    mapping = {c["id"]: c for c in pool}
    trials = []
    predictions = []
    selections = {}

    def evaluate(config, iterations, strategy, stage):
        try:
            pred, evidence = fit_predict(config, iterations, x, y, vx, seed)
        except Exception as exc:
            trials.append(
                {
                    "strategy": strategy,
                    "stage": stage,
                    "candidate": config["id"],
                    "requested_iterations": iterations,
                    "status": "failure",
                    **config,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            raise
        metric = losses(vy, pred, scale * scale)
        trial = {
            "strategy": strategy,
            "stage": stage,
            "candidate": config["id"],
            "requested_iterations": iterations,
            "status": "success",
            **config,
            **evidence,
            **metric,
        }
        trials.append(trial)
        predictions.extend(
            {
                "strategy": strategy,
                "stage": stage,
                "candidate": config["id"],
                "source_row": int(row),
                "actual": float(actual),
                "prediction": float(p),
            }
            for row, actual, p in zip(val.source_row, vy, pred)
        )
        return trial

    order = (
        ["random", "halving"] if SEEDS.index(seed) % 2 == 0 else ["halving", "random"]
    )
    try:
        for strategy in order:
            if strategy == "random":
                last = [evaluate(c, 90, strategy, 0) for c in pool[:6]]
            else:
                first = [evaluate(c, 10, strategy, 0) for c in pool]
                second = [
                    evaluate(mapping[t["candidate"]], 30, strategy, 1)
                    for t in promote(first, 6)
                ]
                last = [
                    evaluate(mapping[t["candidate"]], 90, strategy, 2)
                    for t in promote(second, 2)
                ]
            chosen = promote(last, 1)[0]
            own = [t for t in trials if t["strategy"] == strategy]
            assert sum(t["requested_iterations"] for t in own) == 540
            selections[strategy] = {
                "candidate": chosen["candidate"],
                "config": mapping[chosen["candidate"]],
                "validation_mse": chosen["mse"],
                "search_requested_iterations": 540,
                "search_actual_iterations": sum(t["actual_iterations"] for t in own),
                "search_seconds": sum(t["elapsed_seconds"] for t in own),
            }
        # This file is written before any extra full-pool audit or final-test prediction.
        selection = {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": record["id"],
            "seed": seed,
            "role": record["role"],
            "pool": pool,
            "strategy_order": order,
            "partition_counts": part.value_counts().to_dict(),
            "fit_target_variance": scale * scale,
            "selections": selections,
            "test_evaluated": False,
            "audit_used_for_selection": False,
        }
        write_json(folder / "selection.json", selection)
        audits = [evaluate(c, 90, "audit", 0) for c in pool]
        oracle = promote(audits, 1)[0]
        first_ids = {
            t["candidate"]
            for t in promote(
                [t for t in trials if t["strategy"] == "halving" and t["stage"] == 0], 6
            )
        }
        second_ids = {
            t["candidate"]
            for t in promote(
                [t for t in trials if t["strategy"] == "halving" and t["stage"] == 1], 2
            )
        }
        audit = {
            "oracle_candidate": oracle["candidate"],
            "oracle_validation_mse": oracle["mse"],
            "oracle_survives_10": oracle["candidate"] in first_ids,
            "oracle_survives_30": oracle["candidate"] in second_ids,
            "requested_iterations": sum(t["requested_iterations"] for t in audits),
            "actual_iterations": sum(t["actual_iterations"] for t in audits),
            "elapsed_seconds": sum(t["elapsed_seconds"] for t in audits),
            "scope": "Extra research computation outside both search budgets; never changes frozen selection",
            "regret": {
                strategy: values["validation_mse"] - oracle["mse"]
                for strategy, values in selections.items()
            },
        }
        assert audit["requested_iterations"] == 1620
        write_json(folder / "audit.json", audit)
    except Exception as exc:
        write_json(
            folder / "failure.json",
            {
                "status": "INCOMPLETE",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "completed_trial_count": len(trials),
            },
        )
        raise
    finally:
        write_csv(folder / "trials.csv", trials)
        write_csv(folder / "validation_predictions.csv.gz", predictions)
    print(
        json.dumps(
            {
                "dataset": record["id"],
                "seed": seed,
                "searches": "complete",
                "oracle_survives_10": audit["oracle_survives_10"],
                "seconds": sum(t["elapsed_seconds"] for t in trials),
            }
        ),
        flush=True,
    )


def run_test(record, seed):
    folder = ROOT / f'results/runs/{record["id"]}/{seed}'
    selection = json.loads((folder / "selection.json").read_text())
    assert selection["test_evaluated"] is False
    if (folder / "test_metrics.json").exists():
        raise RuntimeError("Test already exists; preserve it instead of overwriting.")
    frame = load_panel(record)
    part = frame["group_id"].map(lambda x: group_partition(x, seed))
    train = frame.loc[part != "test"]
    test = frame.loc[part == "test"]
    features = record["features"]
    target = record["target"]
    x = train[features].to_numpy(dtype=float)
    y = train[target].to_numpy(dtype=float)
    tx = test[features].to_numpy(dtype=float)
    ty = test[target].to_numpy(dtype=float)
    center, scale = scale_target(y)
    rows = []
    metrics = []
    start = time.perf_counter()
    for strategy in ["random", "halving", "mean"]:
        if strategy == "mean":
            pred = np.full(len(test), center)
            evidence = {
                "elapsed_seconds": 0.0,
                "target_center": center,
                "target_scale": scale,
                "actual_iterations": 0,
            }
            candidate = "outer_training_mean"
        else:
            config = selection["selections"][strategy]["config"]
            candidate = config["id"]
            pred, evidence = fit_predict(config, 90, x, y, tx, seed)
        metrics.append(
            {
                "dataset": record["id"],
                "seed": seed,
                "role": record["role"],
                "strategy": strategy,
                "candidate": candidate,
                "outer_train_n": len(train),
                "outer_train_variance": scale * scale,
                **evidence,
                **losses(ty, pred, scale * scale),
            }
        )
        rows.extend(
            {
                "strategy": strategy,
                "source_row": int(row),
                "group_id": group,
                "actual": float(actual),
                "prediction": float(p),
            }
            for row, group, actual, p in zip(test.source_row, test.group_id, ty, pred)
        )
    write_csv(folder / "test_predictions.csv.gz", rows)
    write_json(
        folder / "test_metrics.json",
        {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "refit_and_prediction_seconds": time.perf_counter() - start,
        },
    )
    print(
        json.dumps(
            {
                "dataset": record["id"],
                "seed": seed,
                "test": "complete",
                "normalized_mse": {m["strategy"]: m["normalized_mse"] for m in metrics},
            }
        ),
        flush=True,
    )


def summarize():
    records = json.loads((ROOT / "data/manifest.json").read_text())
    pairs = []
    all_metrics = []
    for record in records:
        for seed in SEEDS:
            folder = ROOT / f'results/runs/{record["id"]}/{seed}'
            if not (folder / "test_metrics.json").exists():
                continue
            metrics = json.loads((folder / "test_metrics.json").read_text())["metrics"]
            all_metrics.extend(metrics)
            lookup = {m["strategy"]: m for m in metrics}
            selection = json.loads((folder / "selection.json").read_text())
            audit = json.loads((folder / "audit.json").read_text())
            pairs.append(
                {
                    "dataset": record["id"],
                    "name": record["name"],
                    "role": record["role"],
                    "seed": seed,
                    "halving_minus_random_nmse": lookup["halving"]["normalized_mse"]
                    - lookup["random"]["normalized_mse"],
                    "halving_minus_random_mse": lookup["halving"]["mse"]
                    - lookup["random"]["mse"],
                    "random_nmse": lookup["random"]["normalized_mse"],
                    "halving_nmse": lookup["halving"]["normalized_mse"],
                    "mean_nmse": lookup["mean"]["normalized_mse"],
                    "halving_search_seconds": selection["selections"]["halving"][
                        "search_seconds"
                    ],
                    "random_search_seconds": selection["selections"]["random"][
                        "search_seconds"
                    ],
                    "oracle_survives_10": audit["oracle_survives_10"],
                    "oracle_survives_30": audit["oracle_survives_30"],
                    "halving_validation_regret": audit["regret"]["halving"],
                    "random_validation_regret": audit["regret"]["random"],
                    "extra_audit_seconds": audit["elapsed_seconds"],
                }
            )
    write_csv(ROOT / "results/paired_comparisons.csv", pairs)
    write_csv(ROOT / "results/all_test_metrics.csv", all_metrics)
    summary = []
    for record in records:
        own = [p for p in pairs if p["dataset"] == record["id"]]
        if not own:
            continue
        diffs = [p["halving_minus_random_nmse"] for p in own]
        summary.append(
            {
                "dataset": record["id"],
                "name": record["name"],
                "role": record["role"],
                "seed_runs": len(own),
                "halving_test_wins": sum(d < 0 for d in diffs),
                "nmse_difference_median": float(np.median(diffs)),
                "nmse_difference_min": min(diffs),
                "nmse_difference_max": max(diffs),
                "oracle_survives_10": sum(p["oracle_survives_10"] for p in own),
                "oracle_survives_30": sum(p["oracle_survives_30"] for p in own),
                "median_random_nmse": float(np.median([p["random_nmse"] for p in own])),
                "median_halving_nmse": float(
                    np.median([p["halving_nmse"] for p in own])
                ),
                "median_mean_nmse": float(np.median([p["mean_nmse"] for p in own])),
            }
        )
    write_json(
        ROOT / "results/summary.json",
        {
            "datasets": summary,
            "scope": "Conditional overlapping-seed benchmark comparisons, not independent confidence intervals or universal search-strategy superiority",
        },
    )


def main():
    phase, role = sys.argv[1:3]
    assert phase in ["search", "test"] and role in ["development", "held_dataset"]
    records = json.loads((ROOT / "data/manifest.json").read_text())
    for record in records:
        if record["role"] != role:
            continue
        for seed in SEEDS:
            (run_search if phase == "search" else run_test)(record, seed)
    if phase == "test":
        summarize()


if __name__ == "__main__":
    main()
